import subprocess

command = "blender-3.2.2-linux-x64/blender -b -P scripts/blender_script.py -- --object_path /data/hf-objaverse-v1/glbs/000-023/0921bb6998d74a58bfeb35ae12660eb5.glb --output_dir ./views --engine CYCLES --num_images 12 --camera_dist 3"
subprocess.run(command, shell=True)
