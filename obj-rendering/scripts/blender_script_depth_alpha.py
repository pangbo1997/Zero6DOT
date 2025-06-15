"""Blender script to render images of 3D models.

This script is used to render images of 3D models. It takes in a list of paths
to .glb files and renders images of each model. The images are from rotating the
object around the origin. The images are saved to the output directory.

Example usage:
    blender -b -P blender_script.py -- \
        --object_path my_object.glb \
        --output_dir ./views \
        --engine CYCLES \
        --scale 0.8 \
        --num_images 12 \
        --camera_dist 1.2

Here, input_model_paths.json is a json file containing a list of paths to .glb.
"""

import argparse
import math
import os
import random
import sys
import time
import urllib.request
from typing import Tuple
import numpy as np
from mathutils import Matrix

import bpy
from mathutils import Vector


parser = argparse.ArgumentParser()
parser.add_argument(
    "--object_path",
    type=str,
    required=True,
    help="Path to the object file",
)
parser.add_argument("--output_dir", type=str, default="./views-omiobject3d")
parser.add_argument(
    "--engine", type=str, default="BLENDER_EEVEE", choices=["CYCLES", "BLENDER_EEVEE"]
)
parser.add_argument("--num_images", type=int, default=250)
parser.add_argument("--camera_dist", type=float, default=1.5)
parser.add_argument("--no_normal", action='store_true', default=False)
parser.add_argument("--no_depth", action='store_true', default=False)


argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)

context = bpy.context
scene = context.scene
render = scene.render

# DEPTH_FORMAT='OPEN_EXR'
DEPTH_FORMAT='OPEN_EXR'
DEPTH_SCALE = 0.1 # 1.4

render.engine = args.engine
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGBA"
render.resolution_x = 512
render.resolution_y = 512
render.resolution_percentage = 100

scene.cycles.device = "GPU"
scene.cycles.samples = 32
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.cycles.transparent_max_bounces = 3
scene.cycles.transmission_bounces = 3
scene.cycles.filter_width = 0.01
scene.cycles.use_denoising = True
scene.render.film_transparent = True


def sample_point_on_sphere(radius: float) -> Tuple[float, float, float]:
    theta = random.random() * 2 * math.pi
    phi = math.acos(2 * random.random() - 1)
    return (
        radius * math.sin(phi) * math.cos(theta),
        radius * math.sin(phi) * math.sin(theta),
        radius * math.cos(phi),
    )


def add_area_lighting() -> None:
    # delete the default light
    bpy.data.objects["Light"].select_set(True)
    bpy.ops.object.delete()
    # add a new light
    bpy.ops.object.light_add(type="AREA")
    light2 = bpy.data.lights["Area"]
    light2.energy = 50000
    bpy.data.objects["Area"].location[2] = 0.5
    bpy.data.objects["Area"].scale[0] = 100
    bpy.data.objects["Area"].scale[1] = 100
    bpy.data.objects["Area"].scale[2] = 100

def add_sun_lighting() -> None:
    # Make light just directional, disable shadows.
    light = bpy.data.lights['Light']
    light.type = 'SUN'
    light.use_shadow = False
    # Possibly disable specular shading:
    light.specular_factor = 1.0
    light.energy = 10.0

    # Add another light source so stuff facing away from light is not completely dark
    bpy.ops.object.light_add(type='SUN')
    light2 = bpy.data.lights['Sun']
    light2.use_shadow = False
    light2.specular_factor = 1.0
    light2.energy = 1
    bpy.data.objects['Sun'].rotation_euler = bpy.data.objects['Light'].rotation_euler
    bpy.data.objects['Sun'].rotation_euler[0] += 180


def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


def load_object(object_path: str):
    """Loads a 3D model into the scene and returns it."""
    # Get the set of objects before the import
    before_import = set(bpy.context.scene.objects)

    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    elif object_path.endswith(".obj"):
        bpy.ops.import_scene.obj(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")

    # Get the set of objects after the import
    after_import = set(bpy.context.scene.objects)

    # The new object(s) is/are the one(s) in 'after_import' but not 'before_import'
    new_objects = list(after_import - before_import)

    if len(new_objects) > 1:
        print("Warning: More than one new object imported. Taking the first one.")

    return new_objects[0]  # return the first new object only


def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def normalize_scene():
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")
    return np.array([*offset, scale])


def setup_camera():
    cam = scene.objects["Camera"]
    cam.location = (0, 1.2, 0)
    cam.data.lens = 35
    cam.data.sensor_width = 32
    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"
    return cam, cam_constraint

def rotation_matrix(axis, theta):
    axis = np.asarray(axis)
    theta = np.radians(theta)
    axis = axis / np.linalg.norm(axis)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    return np.array([
        [a*a + b*b - c*c - d*d, 2*(b*c - a*d), 2*(b*d + a*c), 0],
        [2*(b*c + a*d), a*a + c*c - b*b - d*d, 2*(c*d - a*b), 0],
        [2*(b*d - a*c), 2*(c*d + a*b), a*a + d*d - b*b - c*c, 0],
        [0, 0, 0, 1]
    ])

def save_images(object_file: str) -> None:
    """Saves rendered images of the object in the scene."""
    os.makedirs(args.output_dir, exist_ok=True)
    reset_scene()
    # load the object
    obj = load_object(object_file)
    # object_uid = os.path.basename(object_file).split(".")[0]
    object_uid = object_file.split("/")[-3]
    scale = normalize_scene()
    add_area_lighting()
    cam, cam_constraint = setup_camera()
    # create an empty object to track
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty
    os.makedirs(os.path.join(args.output_dir, object_uid, 'pose'), exist_ok=True)
    intr_path = os.path.join(args.output_dir, object_uid, 'intrinsics.txt')
    scale_path = os.path.join(args.output_dir, object_uid, 'params.txt')
    np.savetxt(scale_path,scale)
    #
    camera = bpy.context.scene.camera

    #
    render = bpy.context.scene.render

    #
    # Set up rendering of depth map.
    scene.use_nodes = True
    if not args.no_normal:
        scene.view_layers["ViewLayer"].use_pass_normal = True
    scene.view_layers["ViewLayer"].use_pass_diffuse_color = True
    scene.view_layers["ViewLayer"].use_pass_object_index = True
    if not args.no_depth:
        scene.view_layers["ViewLayer"].use_pass_z = True
    tree = bpy.context.scene.node_tree
    nodes = tree.nodes
    links = tree.links
    # Clear default nodes
    for n in nodes:
        nodes.remove(n)
    # Create input render layer node.
    render_layers = tree.nodes.new('CompositorNodeRLayers')

    depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.label = 'Depth Output'
    depth_file_output.file_slots[0].use_node_format = True
    depth_file_output.format.file_format = DEPTH_FORMAT
    # depth_file_output.format.color_depth = str(COLOR_DEPTH)
    if DEPTH_FORMAT == 'OPEN_EXR':
        links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
    else:
        depth_file_output.format.color_mode = "BW"
        depth_file_output.format.file_format = 'PNG'
        depth_file_output.format.color_depth = '16'
        # Remap as other types can not represent the full range of depth.
        depthmap = nodes.new(type="CompositorNode+MapValue")
        # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
        depthmap.offset = [-0.7]
        depthmap.size = [DEPTH_SCALE]
        depthmap.use_min = True
        depthmap.min = [0]
        links.new(render_layers.outputs['Depth'], depthmap.inputs[0])
        links.new(depthmap.outputs[0], depth_file_output.inputs[0])

    normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    normal_file_output.label = 'Normal Output'
    links.new(render_layers.outputs['Normal'], normal_file_output.inputs[0])
    for output_node in [depth_file_output, normal_file_output]:
            output_node.base_path = ''

    # 计算内参
    focal_length = camera.data.lens
    sensor_width = camera.data.sensor_width
    width = render.resolution_x * render.resolution_percentage / 100
    height = render.resolution_y * render.resolution_percentage / 100
    px = width / 2.0
    py = height / 2.0
    fx = width * focal_length / sensor_width
    fy = fx  # 假设焦距在x和y方向上相同

    #
    # content = f"{fx} {px} {py} 0.\n0. 0. 0.\n1.\n{int(width)} {int(height)}"
    content = f"{fx} 0. {px} 0. \n0. {fy} {py} 0.\n0. 0. 1. 0.\n0. 0. 0. 1."

    #
    with open(intr_path, 'w') as file:
        file.write(content)

    print(f"Camera intrinsic parameters written to {intr_path}")

    # Initiate
    previous_theta = random.uniform(0, math.pi * 2)
    previous_phi = random.uniform(math.pi * 0.2, math.pi * 0.6)
    previous_offsets = [random.uniform(-0.2, 0.2) for _ in range(3)]
    previous_offsets[2] = random.uniform(-1, -0.2)
    # Define a velocity
    theta_velocity = random.uniform(-0.008,0.008)
    phi_velocity = random.uniform(-0.015,0.015)

    # offsets
    offset_velocity = [random.uniform(-0.0002, 0.0002) for _ in range(3)]
    frames_per_sequence = np.random.randint(1, 8)

    for i in range(args.num_images):
        # random acceleration
        if i % frames_per_sequence == 0:
            frames_per_sequence = np.random.randint(1, 8)
            theta_acceleration = random.uniform(-0.008, 0.008)
            phi_acceleration = random.uniform(-0.015, 0.015) - 0.02 * (previous_phi - math.pi / 3) / frames_per_sequence  # feedback to keep phi around pi / 2
            offset_acceleration = [random.uniform(-0.0001, 0.0001) for _ in range(3)]

            # theta_velocity += theta_acceleration
            # phi_velocity += phi_acceleration
            offset_velocity = [offset_velocity[j] + offset_acceleration[j] for j in range(3)]

            if abs(theta_velocity) > 0.03:
                theta_velocity = theta_velocity - abs(theta_acceleration) if theta_velocity > 0 else theta_velocity + abs(theta_acceleration)
            else:
                theta_velocity += theta_velocity

            if abs(phi_velocity) > 0.05:
                phi_velocity = phi_velocity - abs(phi_acceleration) if phi_velocity > 0 else phi_velocity + abs(phi_acceleration)
            else:
                if previous_phi > math.pi * 0.6:
                    phi_velocity -= abs(phi_acceleration)
                elif previous_phi < math.pi * 0.1:
                    phi_velocity += abs(phi_acceleration)
                else:
                    phi_velocity += phi_acceleration


        theta = (previous_theta + theta_velocity) % (2 * math.pi)
        phi = min(max(previous_phi + phi_velocity, math.pi * 0.1), math.pi * 0.7)

        previous_theta, previous_phi = theta, phi


        offsets = [(previous_offsets[j] + offset_velocity[j]) for j in range(3)]

        # clip the offsets into [-0.2, 0.2]
        offsets = [min(max(offset,-0.001),0.001) for offset in offsets]

        previous_offsets = offsets

        point = (
            args.camera_dist * math.sin(phi) * math.cos(theta) + offsets[0],
            args.camera_dist * math.sin(phi) * math.sin(theta) + offsets[1],
            args.camera_dist * math.cos(phi) + offsets[2]
        )
        cam.location = point
        empty.location += Vector(offsets)


        # render the image
        scene.frame_set(i)
        render_path = os.path.join(args.output_dir, object_uid, 'color', f"{i}.png")
        scene.render.filepath = render_path
        # depth_file_output.file_slots[0].path = scene.render.filepath.replace('rgb', 'depth')
        # normal_file_output.file_slots[0].path = scene.render.filepath.replace('rgb', 'normal')
        depth_file_output.file_slots[0].path = os.path.join(args.output_dir, object_uid, 'depth/')
        normal_file_output.file_slots[0].path = os.path.join(args.output_dir, object_uid, 'normal/')

        if obj.type == 'MESH' and obj.data.materials:
            for slot in obj.material_slots:
                mat = slot.material

                # 主要只调整 Principled BSDF
                if mat.node_tree:
                    node = mat.node_tree.nodes.get('Principled BSDF')

                    if node:

                        # node.inputs['Base Color'].default_value = (1, 0, 0, 1)
                        node.inputs['Roughness'].default_value = 0.8
                        node.inputs['Specular'].default_value = 0.2


        bpy.ops.render.render(write_still=True)

        matrix = np.array(cam.matrix_world)

        rotation_matrix_x = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])


        matrix = np.dot(matrix, rotation_matrix_x)
        rot_np = matrix[:3, :3]

        for u in range(3):
            col_vec = rot_np[:, u]
            norm_vec = col_vec / np.linalg.norm(col_vec)
            rot_np[:, u] = norm_vec

        matrix[:3, :3] = rot_np
        # pose = matrix.reshape(-1)


        pose_path = os.path.join(args.output_dir, object_uid, 'pose', f"{i}.txt")

        # with open(pose_path, 'w') as file:
        #     file.write(' '.join(map(str, pose)))
        np.savetxt(pose_path, matrix)


def download_object(object_url: str) -> str:
    """Download the object and return the path."""
    # uid = uuid.uuid4()
    uid = object_url.split("/")[-1].split(".")[0]
    tmp_local_path = os.path.join("tmp-objects", f"{uid}.glb" + ".tmp")
    local_path = os.path.join("tmp-objects", f"{uid}.glb")
    # wget the file and put it in local_path
    os.makedirs(os.path.dirname(tmp_local_path), exist_ok=True)
    urllib.request.urlretrieve(object_url, tmp_local_path)
    os.rename(tmp_local_path, local_path)
    # get the absolute path
    local_path = os.path.abspath(local_path)
    return local_path


if __name__ == "__main__":
    try:
        start_i = time.time()
        if args.object_path.startswith("http"):
            local_path = download_object(args.object_path)
        else:
            local_path = args.object_path
        save_images(local_path)
        end_i = time.time()
        print("Finished", local_path, "in", end_i - start_i, "seconds")
        # delete the object if it was downloaded
        if args.object_path.startswith("http"):
            os.remove(local_path)
    except Exception as e:
        print("Failed to render", args.object_path)
        print(e)
