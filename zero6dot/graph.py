from typing import Any
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["CUDA_LAUNCH_BLOCKING"]="1"
import open3d as o3d
import matplotlib.colors as mcolors
import random
import copy
import numpy as np
import time
import torchvision
import open3d as o3d
import g2o
import cv2

from scipy.spatial.distance import cdist

from utils import *
from scipy.spatial.distance import cdist
import os
from third_party.SuperGlobal.model.CVNet_Rerank_model import CVNet_Rerank
from third_party.SuperGlobal.core import transforms
import cv2
import torch
import sys
import numpy as np

_MEAN = [0.406, 0.456, 0.485]
_SD = [0.225, 0.224, 0.229]
from scipy.spatial.transform import Rotation
def prepare(x):
    x=x/255.0
    x=transforms.color_norm(x, _MEAN, _SD)
    # x=torch.tensor(x,dtype=torch.float32)
    return x

class Point:
    def __init__(self,optimizer,vertex_id,pts_xyz,color,fixed=False):
        self.optimizer=optimizer
        self.vertex_id=vertex_id
        self.pts_xyz=pts_xyz
        self.color=color
        self.fixed=fixed
        self.covariance=np.eye(3)*0.5
        self.age=0



class Frame:
    def __init__(self,optimizer,vertex_id,pose,descriptors,pts_vertex_ids,fixed=False):
        self.optimizer=optimizer
        self.vertex_id=vertex_id
        self.pose=pose
        self.fixed=fixed
        self.age=0
        self.descirptors=descriptors
        self.pts_vertex_ids=pts_vertex_ids

        
def project_to_image(feature_position, camera_matrix, robot_pose_matrix):
    feature_position_homogeneous = np.array([feature_position[0], feature_position[1], feature_position[2], 1])
    feature_in_camera = robot_pose_matrix @ feature_position_homogeneous
    feature_in_camera /= feature_in_camera[3]
    feature_in_image_homogeneous = camera_matrix @ feature_in_camera[:3]
    u = feature_in_image_homogeneous[0] / feature_in_image_homogeneous[2]
    v = feature_in_image_homogeneous[1] / feature_in_image_homogeneous[2]
    return np.array([u, v])

def compute_jacobian(feature_position, camera_matrix, robot_pose_matrix):
    X_c, Y_c, Z_c, _ = robot_pose_matrix @ np.array([feature_position[0], feature_position[1], feature_position[2], 1])
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    H = np.array([
        [fx / Z_c, 0, -fx * X_c / (Z_c**2)],
        [0, fy / Z_c, -fy * Y_c / (Z_c**2)]
    ])@robot_pose_matrix[:3,:3]
    return H

def ekf_update(feature, observation, robot_pose_matrix, camera_matrix, observation_noise_covariance):
    observation=np.array(observation)
    predicted_observation = project_to_image(feature.pts_xyz, camera_matrix, robot_pose_matrix)
    observation_residual = observation - predicted_observation
    H = compute_jacobian(feature.pts_xyz, camera_matrix, robot_pose_matrix)
    S = H @ feature.covariance @ H.T + observation_noise_covariance
    K = feature.covariance  @ H.T @ np.linalg.inv(S)
    feature.pts_xyz = feature.pts_xyz+ (K @ observation_residual)[:3]
    feature.covariance  = (np.eye(3) - K @ H) @ feature.covariance 

class Graph:
    def __init__(self,args,video):

        self.video=video 

        self.pts_id_dict={}
        self.frame_id_dict={}
        self.edge_dict={}

        self.args=args
        self.optimizer = g2o.SparseOptimizer()
        self.solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
        self.solver = g2o.OptimizationAlgorithmLevenberg(self.solver)
        self.optimizer.set_algorithm(self.solver)

        self.pts_age=None
        self.frame_id_offset=1000000
        self.fist_frame_id=self.frame_id_offset
        self.last_fragment=None

        self.superglobal = CVNet_Rerank(50, 2048, True)
        weight = torch.load('third_party/SuperGlobal/weights/CVPR2022_CVNet_R50.pyth')
        weight_new = {}
        for i,j in zip(weight['model_state'].keys(), weight['model_state'].values()):
                weight_new[i.replace('globalmodel','encoder_q')] = j     
        mis_key = self.superglobal.load_state_dict(weight_new, strict=False)
        self.superglobal.cuda()
        self.image_global_f=torch.tensor([],dtype=torch.float32)
        self.descriptor=torch.tensor([],dtype=torch.float32).cuda()
        self.gap=20
        self.need_reinit=False

    def get_points(self,vertex_ids=None,return_id=False):
        pts_3d=[]
        pts_color=[]

        pts_history=[]
        pts_ids=[]
        if vertex_ids is None:
            vertex_ids=self.pts_id_dict.keys()
        
        for pts_id in vertex_ids:
            if self.optimizer.vertex(pts_id) is not None:
                pts_3d.append(self.optimizer.vertex(pts_id).estimate())
                pts_color.append(self.pts_id_dict[pts_id].color[[2,1,0]])
                pts_ids.append(pts_id)

        pts_3d=np.stack(pts_3d)
        pts_color=np.stack(pts_color)
        save_pts=o3d.geometry.PointCloud()
        save_pts.points=o3d.utility.Vector3dVector(pts_3d)
        save_pts.colors=o3d.utility.Vector3dVector(pts_color/255.0)
        
        if return_id:
            return save_pts,pts_ids
        else:
            return save_pts
        
    def update_graph(self,fragment,reinit_RT=None):
        RT=np.diag(np.ones(4))
        if fragment.fragment_id==0 or reinit_RT is not None:  
            if reinit_RT is not None:
                RT=reinit_RT
            else:
                if 'init' in self.args.imagedir:
                    RT=np.loadtxt(os.path.join(self.args.data_base_dir,'poses_init/0.txt'))
                    RT_10=np.loadtxt(os.path.join(self.args.data_base_dir,'poses_init/10.txt'))
                else:
                    RT=np.loadtxt(os.path.join(self.args.data_base_dir,'poses/0.txt'))
                    RT_10=np.loadtxt(os.path.join(self.args.data_base_dir,'poses/10.txt'))
            


            pts1=fragment.track[0].cpu().numpy()
            intri=self.video.intrinsics[0].cpu().numpy()
            K1=np.diag(np.ones(3))
            K1[0,0],K1[1,1],K1[0,2],K1[1,2]=intri[0],intri[1],intri[2],intri[3] 
            pts1_norm = cv2.undistortPoints(np.expand_dims(pts1, axis=1), cameraMatrix=K1, distCoeffs=None)

            pts2=fragment.track[10].cpu().numpy()
            intri=self.video.intrinsics[10].cpu().numpy()
            K10=np.diag(np.ones(3))
            K10[0,0],K10[1,1],K10[0,2],K10[1,2]=intri[0],intri[1],intri[2],intri[3]                                
            pts10_norm = cv2.undistortPoints(np.expand_dims(pts2, axis=1), cameraMatrix=K10, distCoeffs=None)

            P1 = np.dot(K1, RT[:3])
            P2 = np.dot(K10, RT_10[:3])

            points_4d_hom = cv2.triangulatePoints(P1, P2, pts1_norm, pts10_norm)

            pts = (points_4d_hom[:3] / points_4d_hom[3]).transpose(1,0)
            # import pdb;pdb.set_trace()
            us,vs=fragment.track[0,:,0].cpu().numpy(),fragment.track[0,:,1].cpu().numpy()

        else:
            us,vs=fragment.track[0,:,0].cpu().numpy(),fragment.track[0,:,1].cpu().numpy()
            pose=self.optimizer.vertex(self.frame_id_offset-1).estimate()
            t=pose.translation()
            R=pose.rotation().matrix()
            RT[:3,:3]=R
            RT[:3,3]=t
            pts_array=np.array(self.get_points(self.last_fragment.vertex_ids).points)
            pts_array_uv=(pts_array@R.T+t)
            fx,fy,cx,cy=fragment.intrinsics[0].cpu().numpy()
            x=pts_array_uv[:,0]
            y=pts_array_uv[:,1]
            z=pts_array_uv[:,2]
            u_corrd=((x*fx/z)+cx)
            v_corrd=((y*fy/z)+cy)
            uv_coord=np.concatenate((u_corrd[:,None],v_corrd[:,None]),axis=1)
            uv=np.concatenate((us[:,None],vs[:,None]),axis=1)
            pts=pts_array[cdist(uv,uv_coord).argmin(axis=1)]

        for i in range(fragment.track.shape[0]):
            if i>=10 and fragment.fragment_id==0: RT=RT_10
            self.frame_id_dict[self.frame_id_offset+i]=Frame(self.optimizer,self.frame_id_offset+i,RT,fragment.descriptors,fragment.vertex_ids)



        for i in range(len(fragment.vertex_ids)):
            if not fragment.vertex_ids[i] in self.pts_id_dict.keys():
                # import pdb;pdb.set_trace()
                v=min(vs[i].astype(np.int32),fragment.images.shape[2])
                u=min(us[i].astype(np.int32),fragment.images.shape[3])
                color=fragment.images[0,:,v,u].detach().cpu().numpy()
                self.pts_id_dict[fragment.vertex_ids[i]]=Point(self.optimizer,fragment.vertex_ids[i],pts[i],color)
                

        for i in range(fragment.track.shape[0]):
            for j in range(len(fragment.vertex_ids)):
                # import pdb;pdb.set_trace()
                if fragment.visible[i][j]!=0:
                    self.edge_dict[(self.frame_id_offset+i,fragment.vertex_ids[j])]=(fragment.track[i,j],fragment.intrinsics[i].cpu().numpy())
        
        self.last_fragment=fragment
        
    def update_optimizer(self,fragment):
        for j in range(len(fragment.vertex_ids)):
            last_i=0
            for i in range(fragment.track.shape[0]):
                if fragment.fragment_id==0 and self.optimizer.vertex(self.frame_id_offset+i) is None:
                    RT=self.frame_id_dict[self.frame_id_offset+i].pose
                    pose = g2o.SE3Quat(RT[:3,:3], RT[:3,3]) 
                    v_se3 = g2o.VertexSE3Expmap()
                    v_se3.set_id(self.frame_id_offset+i)
                    v_se3.set_estimate(pose)
                    self.optimizer.add_vertex(v_se3)
                else:
                    v_se3=self.optimizer.vertex(self.frame_id_offset+i)

                if fragment.visible[i][j]:
                    last_i=i
                    if self.optimizer.vertex(fragment.vertex_ids[j]) is None:
                        pts=self.pts_id_dict[fragment.vertex_ids[j]].pts_xyz
                        vp = g2o.VertexSBAPointXYZ()
                        vp.set_id(fragment.vertex_ids[j])
                        vp.set_marginalized(True)
                        vp.set_estimate(pts)
                        self.optimizer.add_vertex(vp)
                    else:
                        vp=self.optimizer.vertex(fragment.vertex_ids[j])
                    
                    edge = g2o.EdgeSE3ProjectXYZ()
                    edge.set_vertex(0, vp)
                    edge.set_vertex(1, v_se3)
                    edge.set_measurement(self.edge_dict[(self.frame_id_offset+i,fragment.vertex_ids[j])][0])
                    edge.set_information(np.identity(2))
                    edge.set_robust_kernel(g2o.RobustKernelHuber())
                    edge.set_parameter_id(0, 0)
                    edge.fx=fragment.intrinsics[i][0].cpu().numpy()
                    edge.fy=fragment.intrinsics[i][1].cpu().numpy()
                    edge.cx=fragment.intrinsics[i][2].cpu().numpy()
                    edge.cy=fragment.intrinsics[i][3].cpu().numpy()
                    self.optimizer.add_edge(edge)
        

            

        if self.last_fragment.fragment_id==0:
            self.optimizer.vertex(self.frame_id_offset).set_fixed(True)
            self.optimizer.vertex(self.frame_id_offset+10).set_fixed(True)
                    
        # if self.last_fragment.fragment_id==0:
        #     for i in range(self.gap):
        #         if self.optimizer.vertex(self.frame_id_offset+i) is not None:
        #             self.optimizer.vertex(self.frame_id_offset+i).set_fixed(True)


    def update_offset(self,length):
        self.frame_id_offset+=length

    def fix_keyframe_and_points(self):

        for pts_id in self.pts_id_dict.keys():
            if self.optimizer.vertex(pts_id) is not None:
                if self.pts_id_dict[pts_id].age>=2:
                    self.optimizer.vertex(pts_id).set_fixed(True)

        if self.frame_id_offset-self.fist_frame_id<=20: 
            for frame_id in range(self.frame_id_offset-self.last_fragment.track.shape[0],self.frame_id_offset):
                self.optimizer.vertex(frame_id).set_fixed(True)
            return
        
        for frame_id in range(self.frame_id_offset-self.last_fragment.track.shape[0],self.frame_id_offset):

            if (frame_id+1)%10==0 or frame_id==self.fist_frame_id:
                self.optimizer.vertex(frame_id).set_fixed(True)
            else:
                self.optimizer.remove_vertex(self.optimizer.vertex(frame_id))


    def update_global(self):
        for frame_id in self.frame_id_dict.keys():
            if self.optimizer.vertex(frame_id) is not None and frame_id-self.fist_frame_id>20:
                self.optimizer.vertex(frame_id).set_fixed(False)
        for pts_id in self.pts_id_dict.keys():
            if self.optimizer.vertex(pts_id) is not None:
                self.optimizer.vertex(pts_id).set_fixed(False)
        
        self.optimizer.initialize_optimization()
        self.optimizer.optimize(15)     

        for frame_id in self.frame_id_dict.keys():
            if self.optimizer.vertex(frame_id) is not None and frame_id-self.fist_frame_id>20:
                self.optimizer.vertex(frame_id).set_fixed(True)
        for pts_id in self.pts_id_dict.keys():
            if self.optimizer.vertex(pts_id) is not None and self.pts_id_dict[pts_id].age>=2:
                self.optimizer.vertex(pts_id).set_fixed(True)

    def remove_bad_points(self):

        pts,pts_id=self.get_points(return_id=True)
        pose=self.optimizer.vertex(self.frame_id_offset-1).estimate()
        t=pose.translation()
        R=pose.rotation().matrix()
        RT=np.zeros((3,4))
        RT[:3,:3]=R
        RT[:3,3]=t
        pts_array=(np.array(pts.points))@R.T+t
        fx,fy,cx,cy=self.last_fragment.intrinsics[-1].cpu().numpy()
        K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
        img=self.last_fragment.images[-1]
        pos1=np.stack(get_uvcoord(pts_array,K,img.shape[2],img.shape[1])).transpose(1,0)
        
        for i in range(pos1.shape[0]):
            if pos1[i][0]<img.shape[1] and pos1[i][1]<img.shape[2]:
                # if self.last_fragment.fragment_id==3 and i==15: import pdb;pdb.set_trace()
                if  img[:,pos1[i][0]-10:pos1[i][0]+10,pos1[i][1]-10:pos1[i][1]+10].sum().item()==0:
                    self.optimizer.remove_vertex(self.optimizer.vertex(pts_id[i]))

        fragment=self.last_fragment
        for i in range(fragment.track.shape[0]):
            pose=self.optimizer.vertex(self.frame_id_offset-fragment.track.shape[0]+i).estimate()
            t=pose.translation()
            R=pose.rotation().matrix()
            fx,fy,cx,cy=self.last_fragment.intrinsics[i].cpu().numpy()
            K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
            for j in range(len(fragment.vertex_ids)):
                if fragment.visible[i][j]:
                    if self.optimizer.vertex(fragment.vertex_ids[j]) is not None:


                        pts_3d=self.optimizer.vertex(fragment.vertex_ids[j]).estimate()
                        pts_2d=fragment.track[i][j]

                        pts_array=(pts_3d[None])@R.T+t
                        repoj_pos=np.stack(get_uvcoord(pts_array,K,img.shape[2],img.shape[1])).transpose(1,0)[0,[1,0]]
                        if np.linalg.norm(pts_2d-repoj_pos)>5:
                            self.optimizer.remove_vertex(self.optimizer.vertex(fragment.vertex_ids[j]))


    def collect_img_global_feature(self,fragment):
        curr_img_f=self.superglobal.extract_global_descriptor(prepare(fragment.images[0])[None],True,True,True,3).detach().cpu()
        self.image_global_f=torch.cat((self.image_global_f,curr_img_f)).detach().cpu()
        self.curr_img_f=curr_img_f

    def loop_closing(self):
        eps=1e-6
        start=-1
        end=-1
        if self.last_fragment.fragment_id<=5: return
        curr_img_f=self.curr_img_f
        sim_matrix=(curr_img_f@self.image_global_f[:-5].T)[0]
        max_sim_index=sim_matrix.argmax()
        start=self.fist_frame_id+(max_sim_index.item())*self.gap
        end=self.fist_frame_id+(self.last_fragment.fragment_id)*self.gap

        if sim_matrix[max_sim_index]>=0.95:


            # print(start,end,sim_matrix[max_sim_index])
            print('loop closing')

            point_iter_idxs=self.frame_id_dict[start].pts_vertex_ids
            point_curr_idxs=self.frame_id_dict[end].pts_vertex_ids


            desc_iter=self.frame_id_dict[start].descirptors
            desc_curr=self.frame_id_dict[end].descirptors

            bf = cv2.BFMatcher(cv2.NORM_L2)
            matches = bf.match(desc_curr.detach().cpu().numpy(),desc_iter.detach().cpu().numpy())
            desc_curr_idx = np.array([m.queryIdx for m in matches])
            desc_iter_idx = np.array([m.trainIdx for m in matches])
            # import pdb;pdb.set_trace()
            for src_idx,dst_idx in zip(desc_iter_idx,desc_curr_idx):

                point_iter_idx=point_iter_idxs[src_idx]
                point_curr_idx=point_curr_idxs[dst_idx]

                if point_curr_idx in point_iter_idxs: continue

                if self.optimizer.vertex(point_iter_idx) is not None and self.optimizer.vertex(point_curr_idx) is not None:
                    self.optimizer.vertex(point_curr_idx).set_estimate(self.optimizer.vertex(point_iter_idx).estimate())


                    for frame_idx in range(self.fist_frame_id,self.frame_id_offset):         
                        if self.optimizer.vertex(frame_idx) is not None:
                            if (frame_idx,point_iter_idx) in self.edge_dict.keys(): 
                                edge = g2o.EdgeSE3ProjectXYZ()  
                                edge.set_vertex(0, self.optimizer.vertex(point_curr_idx))
                                edge.set_vertex(1, self.optimizer.vertex(frame_idx))
                                edge.set_measurement(self.edge_dict[(frame_idx,point_iter_idx)][0])
                                fx,fy,cx,cy=self.edge_dict[(frame_idx,point_iter_idx)][1]
                                edge.set_information(np.identity(2))
                                edge.set_robust_kernel(g2o.RobustKernelHuber())
                                edge.set_parameter_id(0, 0)
                                self.optimizer.add_edge(edge)
                                edge.fx=fx
                                edge.fy=fy
                                edge.cx=cx
                                edge.cy=cy
                                self.edge_dict[(frame_idx,point_curr_idx)]=(self.edge_dict[(frame_idx,point_iter_idx)][0],self.edge_dict[(frame_idx,point_iter_idx)][1])
                    
                    if self.optimizer.vertex(point_iter_idx) is not None:
                        self.optimizer.remove_vertex(self.optimizer.vertex(point_iter_idx))

            self.optimizer.initialize_optimization()
            self.optimizer.optimize(20)


    def reinit(self,fragment):

        curr_img_f=self.curr_img_f
        sim_matrix=(curr_img_f@self.image_global_f[:-3].T)[0]
        max_sim_index=sim_matrix.argmax()
        start=self.fist_frame_id+(max_sim_index.item())*self.gap
        end=self.fist_frame_id+(fragment.fragment_id)*self.gap
        if sim_matrix[max_sim_index]>=0.95:
            print(start,end,sim_matrix[max_sim_index])
            print('re-init')
            point_iter_idxs=self.frame_id_dict[start].pts_vertex_ids
            desc_iter=[]
            pts_3d=[]
            for i,pts_id in enumerate(point_iter_idxs):
                if self.optimizer.vertex(pts_id) is not None:
                    pts_3d.append(self.optimizer.vertex(pts_id).estimate())
                    desc_iter.append(self.frame_id_dict[start].descirptors[i])
            pts_3d=np.stack(pts_3d)
            desc_iter=torch.stack(desc_iter)

            desc_curr=fragment.descriptors
            bf = cv2.BFMatcher(cv2.NORM_L2)
            matches = bf.match(desc_curr.detach().cpu().numpy(),desc_iter.detach().cpu().numpy())
            desc_curr_idx = np.array([m.queryIdx for m in matches])
            desc_iter_idx = np.array([m.trainIdx for m in matches])

            _,rvec,tvec,_=cv2.solvePnPRansac(pts_3d[desc_iter_idx],fragment.track[0][desc_curr_idx].cpu().numpy(),self.K,None)

            RT=np.zeros((4,4))
            RT[:3,:3]=Rotation.from_rotvec(rvec[:,0]).as_matrix()
            RT[:3,3]=tvec[:,0]
            RT[3,3]=1

            self.update_graph(fragment,reinit_RT=RT)
            self.update_optimizer(fragment)


            for pts_id in self.pts_id_dict.keys():
                if self.pts_id_dict[pts_id] is not None:
                    self.pts_id_dict[pts_id].age+=1


            point_iter_idxs=self.frame_id_dict[start].pts_vertex_ids
            point_curr_idxs=self.frame_id_dict[end].pts_vertex_ids


            desc_iter=self.frame_id_dict[start].descirptors
            desc_curr=self.frame_id_dict[end].descirptors


            bf = cv2.BFMatcher(cv2.NORM_L2)
            matches = bf.match(desc_curr.detach().cpu().numpy(),desc_iter.detach().cpu().numpy())
            desc_curr_idx = np.array([m.queryIdx for m in matches])
            desc_iter_idx = np.array([m.trainIdx for m in matches])

            for src_idx,dst_idx in zip(desc_iter_idx,desc_curr_idx):

                point_iter_idx=point_iter_idxs[src_idx]
                point_curr_idx=point_curr_idxs[dst_idx]

                if point_curr_idx in point_iter_idxs: continue

                if self.optimizer.vertex(point_iter_idx) is not None and self.optimizer.vertex(point_curr_idx) is not None:
                    self.optimizer.vertex(point_curr_idx).set_estimate(self.optimizer.vertex(point_iter_idx).estimate())


                    for frame_idx in range(self.fist_frame_id,self.frame_id_offset):         
                        if self.optimizer.vertex(frame_idx) is not None:
                            if (frame_idx,point_iter_idx) in self.edge_dict.keys(): 
                                edge = g2o.EdgeProjectXYZ2UV()  
                                edge.set_vertex(0, self.optimizer.vertex(point_curr_idx))
                                edge.set_vertex(1, self.optimizer.vertex(frame_idx))
                                edge.set_measurement(self.edge_dict[(frame_idx,point_iter_idx)])
                                edge.set_information(np.identity(2))
                                edge.set_robust_kernel(g2o.RobustKernelHuber())
                                edge.set_parameter_id(0, 0)
                                self.optimizer.add_edge(edge)
                                self.edge_dict[(frame_idx,point_curr_idx)]=self.edge_dict[(frame_idx,point_iter_idx)]
                    
                    if self.optimizer.vertex(point_iter_idx) is not None:
                        self.optimizer.remove_vertex(self.optimizer.vertex(point_iter_idx))

            self.optimizer.initialize_optimization()
            self.optimizer.optimize(20)
            return True
        else:
            return False

    def frontend(self,fragment):
        just_sucess_reinit=False
        if fragment.need_reinit:
            self.need_reinit=True
            return 
        if self.need_reinit:
            just_sucess_reinit=self.reinit(fragment)
            if just_sucess_reinit: self.need_reinit=False

        if not self.need_reinit and not just_sucess_reinit:


            if fragment.fragment_id!=0:
                self.update_graph(fragment)

                for vertex_id in self.pts_id_dict.keys():
                    if self.optimizer.vertex(vertex_id) is not None:
                        self.pts_id_dict[vertex_id].pts_xyz=self.optimizer.vertex(vertex_id).estimate()

                for i in range(fragment.track.shape[0]):

                    RT=self.frame_id_dict[self.frame_id_offset+i].pose
                    pose = g2o.SE3Quat(RT[:3,:3], RT[:3,3]) 
                    v_se3 = g2o.VertexSE3Expmap()
                    v_se3.set_id(self.frame_id_offset+i)
                    v_se3.set_estimate(pose)
                    self.optimizer.add_vertex(v_se3)



                for j in range(len(fragment.vertex_ids)):
                    if self.optimizer.vertex(fragment.vertex_ids[j]) is not None:
                        self.optimizer.vertex(fragment.vertex_ids[j]).set_estimate(self.pts_id_dict[fragment.vertex_ids[j]].pts_xyz)


                self.update_optimizer(fragment)
                self.optimizer.initialize_optimization()
                self.optimizer.optimize(20)


            else:

                self.update_graph(fragment)
                self.update_optimizer(fragment)
                self.optimizer.initialize_optimization()
                self.optimizer.optimize(20)


            for pts_id in self.pts_id_dict.keys():
                if self.pts_id_dict[pts_id] is not None:
                    self.pts_id_dict[pts_id].age+=1

    def backend(self):
        if not self.need_reinit:
            self.remove_bad_points()
            self.fix_keyframe_and_points()
            self.loop_closing()
            self.update_global()

        

    



        
        


 