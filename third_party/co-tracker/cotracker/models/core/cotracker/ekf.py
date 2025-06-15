import torch


from pytorch3d.transforms import *


def skew_matrix_from_vector(v):

    _v=torch.zeros(v.shape[0],3,3,device=v.device)
    _v[:,0,1]=-v[:,2]
    _v[:,1,0]=v[:,2]
    _v[:,0,2]=v[:,1]
    _v[:,2,0]=-v[:,1]
    _v[:,1,2]=-v[:,0]
    _v[:,2,1]=v[:,0]

    return _v

def phi(h):
    B=h.shape[0]
    _phi=torch.zeros(B,4,4,device=h.device)
    _phi[:,0,0]=h[:,0]
    _phi[:,0,1:]=-h[:,1:]
    _phi[:,1:,0]=h[:,1:]
    
    I_h=torch.eye(3,device=h.device)[None].repeat(B,1,1)
    I_h[:,0,0]=h[:,0]
    I_h[:,1,1]=h[:,0]
    I_h[:,2,2]=h[:,0]

    _phi[:,1:,1:]=I_h-skew_matrix_from_vector(h[:,1:])

    return _phi


def ekf_predict(x,x_cov,v,w,predict_noise_variance=0.05):
    B=x.shape[0]
    N=x.shape[1]
    x[:,:3]=x[:,:3]+v
    x[:,3:7]=quaternion_multiply(x[:,3:7].detach().clone(),w)
    with torch.no_grad():
        G=torch.eye(x.shape[1],device=x.device)[None].repeat(B,1,1)
        G[:,3:7,3:7]=phi(w.detach().clone())
        x_cov=G@(x_cov)@(G.permute(0,2,1))+predict_noise_variance*torch.eye(N,device=x.device)[None].repeat(B,1,1)
    return x,x_cov


def jacobian_h_pc(pts,K,pose):

    B=pts.shape[0]
    pts=pts.reshape(B,-1,3)
    N=pts.shape[1]
    pts_homo=torch.cat([pts,torch.ones(B,N,1,device=pts.device)],dim=-1)
    X_c, Y_c, Z_c,_ = torch.unbind((pose @ pts_homo[:,:,:,None])[:,:,:,0],dim=-1)
    fx, fy = K[:,0, 0], K[:,1, 1]
    H=torch.zeros(B,N,2,3,device=pts.device)
    H[:,:,0,0]=fx / Z_c
    H[:,:,0,2]=-fx * X_c / (Z_c**2)
    H[:,:,1,1]=fy / Z_c
    H[:,:,1,2]=-fy * Y_c / (Z_c**2)
    return H

def jacobian_h_pw(pts,K,pose):
    return jacobian_h_pc(pts,K,pose)@pose[:,:3,:3]




def jacobian_h_rk(pts,K,pose):
    return jacobian_h_pc(pts,K,pose)

def jacobian_R_q(q):
    qr,qx,qy,qz=torch.unbind(q[:,:,None],dim=1)
    B=q.shape[0]
    dqr=2*torch.cat([
        qr,-qz,qy,
        qz,qr,-qx,
        -qy,qx,qr
    ],dim=1).reshape(B,3,3)

    dqx=2*torch.cat([
        qx,qy,qz,
        qy,-qx,-qr,
        qz,qr,-qx
    ],dim=1).reshape(B,3,3)
    dqy=2*torch.cat([
        -qy,qx,qr,
        qx,qy,qz,
        -qr,qz,-qy
    ],dim=1).reshape(B,3,3)
    dqz=2*torch.cat([
        -qz,-qr,qx,
        qr,-qz,qy,
        qx,qy,qz
    ],dim=1).reshape(B,3,3)
    return dqr,dqx,dqy,dqz
  
def jacobian_h_q(pts,K,q,pose):
    dqr,dqx,dqy,dqz=jacobian_R_q(q)
    # import pdb;pdb.set_trace()
    B=pts.shape[0]
    pts=pts.reshape(B,-1,3)
    pts=pts[:,:,:,None]
    return jacobian_h_pc(pts[:,:,:,0],K,pose)@(torch.cat([dqr@pts,dqx@pts,dqy@pts,dqz@pts],dim=-1))



def ekf_update(x,x_cov,pose,obs_all,cam_K,observe_noise_cov=1):
    with torch.no_grad():
        B,N=x.shape
        len_o=obs_all.shape[1]
        H=torch.zeros(B,len_o*2,7+3*len_o,device=x.device)
        
        # for i in range(len_o):
            # pts=x[:,7+i*3:7+i*3+3]
            # H[:,i*2:i*2+2,:3]=jacobian_h_rk(pts,cam_K,pose)
            # H[:,i*2:i*2+2,3:7]=jacobian_h_q(pts,cam_K,x[:,3:7],pose)
            # H[:,i*2:i*2+2,7+i*3:7+i*3+3]=jacobian_h_pw(pts,cam_K,pose)

        pts=x[:,7:]
        H[:,:,:3]=jacobian_h_rk(pts,cam_K,pose).reshape(B,-1,3)
        H[:,:,3:7]=jacobian_h_q(pts,cam_K,x[:,3:7],pose).reshape(B,-1,4)

        jac_h_pw=jacobian_h_pw(pts,cam_K,pose)
        rows = torch.arange(len_o*2).view(len_o, 2)
        rows = rows.unsqueeze(2).expand(-1, -1, 3).reshape(1, -1).repeat(B, 1)
        cols = torch.arange(len_o*3).view(len_o, 3)
        cols = cols.unsqueeze(1).expand(-1, 2, -1).reshape(1, -1).repeat(B, 1)
        batch_indices = torch.arange(B).view(-1, 1).expand(-1, rows.shape[1])
        H[:,:,7:][batch_indices, rows, cols] = jac_h_pw.view(B, -1)

        S=torch.linalg.inv(H@x_cov@H.permute(0,2,1)+observe_noise_cov*torch.eye(2*len_o,device=x.device)[None].repeat(B,1,1))
        K=x_cov@H.permute(0,2,1)@S
        feature_position_homogeneous = torch.cat([x[:,7:].reshape(B,-1,3), torch.ones(B,len_o,1,device=x.device)],dim=-1)[:,:,:,None]
        feature_in_camera = pose @ feature_position_homogeneous
        feature_in_camera = feature_in_camera/ feature_in_camera[:,:,3:4]
        feature_in_image_homogeneous = cam_K @ feature_in_camera[:,:,:3]
        v = feature_in_image_homogeneous[:,:,0] / feature_in_image_homogeneous[:,:,2]
        u = feature_in_image_homogeneous[:,:,1] / feature_in_image_homogeneous[:,:,2]


    x=x.clone()+(K@(obs_all-torch.cat([v,u],dim=2)).reshape(B,-1)[:,:,None])[:,:,0]
    with torch.no_grad():
        x_cov=(torch.eye(7+3*len_o,device=x.device)[None].repeat(B,1,1)-K@H)@x_cov
        norm_q=torch.norm(x[:,3:7],dim=1)
        
    x[:,3:7]=x[:,3:7]/norm_q[:,None].detach().clone()
    return x,x_cov

import os
import numpy as np
import cv2
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
def depth_to_pointcloud(depth, K):
    vs, us = depth.nonzero()
    zs = depth[vs, us]
    xs = (us - K[0, 2]) * zs / K[0, 0]
    ys = (vs - K[1, 2]) * zs / K[1, 1]
    pts = np.stack([xs, ys, zs], axis=1)
    return pts

def get_uvcoord(point_cloud,depth_cam_matrix):
    fx,fy = depth_cam_matrix[0,0],depth_cam_matrix[1,1]
    cx,cy = depth_cam_matrix[0,2],depth_cam_matrix[1,2]
    x=point_cloud[:,0]
    y=point_cloud[:,1]
    z=point_cloud[:,2]
    w_corrd=((x*fx/z)+cx)#.astype(np.int64)
    h_corrd=((y*fy/z)+cy)#.astype(np.int64)

    return h_corrd,w_corrd,z

def prepare_track_from_depth(data_base_dir,start_ind,seq_name,seq_len):
    K=np.loadtxt(os.path.join(data_base_dir,'intrinsics.txt'))[:3,:3]
    RT_0=np.loadtxt(os.path.join(data_base_dir,f'pose/{start_ind}.txt'))
    depth_0=cv2.imread(os.path.join(data_base_dir,'depth/{:04d}.exr'.format(start_ind)),-1)[:,:,0]
    depth_0[depth_0==65504]=0
    pts_0=depth_to_pointcloud(depth_0,K)
    choose_idx=np.random.permutation(pts_0.shape[0])[:500]
    pts_0=pts_0[choose_idx]
    pts_0_homo=np.concatenate((pts_0,np.ones(pts_0.shape[0])[:,None]),axis=1)
    v_corrd,u_corrd,z=get_uvcoord(pts_0,K)
    track=np.concatenate((v_corrd[:,None],u_corrd[:,None]),axis=1)
    vis=np.ones(track.shape[0]).astype(np.bool_)
    track_list=[]
    vis_list=[]
    RT_list=[torch.tensor(np.linalg.inv(RT_0))]
    track_list.append(track)
    vis_list.append(vis)
    for idx in range(start_ind+1,start_ind+seq_len):
        depth=cv2.imread(os.path.join(data_base_dir,'depth/{:04d}.exr'.format(idx)),-1)[:,:,0]
        depth[depth==65504]=0
        RT=np.loadtxt(os.path.join(data_base_dir,f'pose/{idx}.txt'))
        RT_list.append(torch.tensor(np.linalg.inv(RT)))
        # RT_list.append(torch.tensor(RT))
        pts=((np.linalg.inv(RT)@RT_0@(pts_0_homo.T)).T)[:,:3]
        v_corrd,u_corrd,z=get_uvcoord(pts,K)
        track=np.concatenate((v_corrd[:,None],u_corrd[:,None]),axis=1)
        # import pdb;pdb.set_trace()
        u_valid=track[:,0]<depth.shape[0]
        v_valid=track[:,1]<depth.shape[1]
        
        new_track=track.copy()
        new_track[(1-u_valid).astype(np.bool_),0]=depth.shape[0]-1
        new_track[(1-v_valid).astype(np.bool_),1]=depth.shape[1]-1

        delta_z=abs((depth[new_track[:,0].astype(np.int64),new_track[:,1].astype(np.int64)]-z))
        hist,bins = np.histogram(delta_z,bins=100)
        vis=delta_z<bins[1]

        vis= vis & u_valid & v_valid

        track_list.append(track)
        vis_list.append(vis)

    pts_gt=torch.tensor((((RT_0@(pts_0_homo.T)).T)[:,:3]).astype(np.float32)).to(torch.float32)
    track=torch.tensor(np.stack(track_list)[:,:,[1,0]]).to(torch.float32)
    # import pdb;pdb.set_trace()
    vs,us=track[0,:,0],track[0,:,1]
    zs = torch.ones(len(us))
    xs = (us - K[0, 2]) * zs / K[0, 0]
    ys = (vs - K[1, 2]) * zs / K[1, 1]
    pts = np.stack([xs, ys, zs], axis=1)
    pts= pts@RT_0[:3,:3].T +RT_0[:3,3]
    pts=torch.tensor(pts)
    noise_pts=torch.randn(pts_gt.shape)
    noise_pts[:,0]*=0.005
    noise_pts[:,1]*=0.005
    noise_pts[:,2]*=0.01

    noise_track=torch.randn(track.shape)*0.5
    data={
        'pts':pts_gt+noise_pts,#torch.tensor(pts),
        'pts_gt':pts_gt,
        'pose':torch.stack(RT_list).to(torch.float32),
        'track':track+noise_track,
        'vis':torch.tensor(np.stack(vis_list)),
        'K':torch.tensor(K).to(torch.float32)
    }

    return data
        
def rotmatrix_to_quat(rotation_matrix):
    r00 = rotation_matrix[0, 0]
    r01 = rotation_matrix[0, 1]
    r02 = rotation_matrix[0, 2]
    r10 = rotation_matrix[1, 0]
    r11 = rotation_matrix[1, 1]
    r12 = rotation_matrix[1, 2]
    r20 = rotation_matrix[2, 0]
    r21 = rotation_matrix[2, 1]
    r22 = rotation_matrix[2, 2]

    w = np.sqrt(1 + r00 + r11 + r22) / 2
    x = (r21 - r12) / (4 * w)
    y = (r02 - r20) / (4 * w)
    z = (r10 - r01) / (4 * w)

    return torch.tensor([w, x, y, z])

def prepare_test_data():
    # if not os.path.exists('demo_data.dict'):
    data=prepare_track_from_depth('./apple_001',0,'0000',20)
    # import pdb;pdb.set_trace()
    torch.save(data,'demo_data.dict')

from scipy.spatial.transform import Rotation
import time
if __name__=='__main__':
    # prepare_test_data()

    data=torch.load('demo_data.dict')
    # import pdb;pdb.set_trace()
    data['track']=data['track']
    data['vis']=data['vis']#[:,0:5]
    data['pts']=data['pts']#[0:5]
    data['pts_gt']=data['pts_gt']#[0:5]
    pts=data['pts']
    x=torch.zeros(7+3*len(pts))
    x[:3]=data['pose'][0][:3,3]
    x[3:7]=rotmatrix_to_quat(data['pose'][0][:3,:3])
    x[7:]=pts.reshape(-1)
    x_cov=torch.zeros(len(x),len(x))

    print('Init Point Error:{}'.format( torch.norm(data['pts']-data['pts_gt'],dim=1).mean()))
    eps=1e-6
    pose=torch.eye(4)
    pose[:3,:3]=quat2rotmatrix(x[3:7])
    pose[:3,3]=x[:3]
    gt_pose=data['pose'][0]
    r_loss=(np.trace(pose[:3,:3]@gt_pose[:3,:3].T)-1)/2
    r_loss=np.clip(r_loss,a_min=-1+eps,a_max=1-eps)
    R_error=np.arccos(r_loss)*180/np.pi
    t_error=np.linalg.norm(pose[:3,3]-gt_pose[:3,3])
    print('Init Pose Error: R_error: {} t_error:{}'.format(R_error,t_error))
    start=time.time()
    for i in range(len(data['track'])):

        if i==0:
            v=torch.zeros(3)
            w=torch.zeros(4)
            w[0]=1
            # v=torch.zeros(3)
            # w=torch.tensor([0.99,0,0,0.01])
        else:
            #v=data['pose'][i][:3,3]-data['pose'][i-1][:3,3]
            #w=rotmatrix_to_quat(torch.linalg.inv(data['pose'][i-1][:3,:3])@data['pose'][i][:3,:3])
            v=data['pose'][i][:3,3]-data['pose'][i-1][:3,3]+torch.randn(3)*0.01
            axis=torch.randn(3)
            axis=axis/torch.norm(axis)
            angle=torch.rand(1)*10*np.pi/180
            rotvec=axis*angle
            noise_predict=torch.tensor(Rotation.from_rotvec(rotvec.numpy()).as_matrix()).to(torch.float32)
            # import pdb;pdb.set_trace()
            w=rotmatrix_to_quat(torch.linalg.inv(data['pose'][i-1][:3,:3])@noise_predict@data['pose'][i][:3,:3])#+torch.randn(4)*0.05
        w=w/torch.norm(w)
        # import pdb;pdb.set_trace()
        x,x_cov=ekf_predict(x,x_cov,v,w)
        pose=torch.eye(4)
        pose[:3,:3]=quat2rotmatrix(x[3:7])
        pose[:3,3]=x[:3]

        print('Predict Points Error :{}'.format(torch.norm(x[7:].reshape(-1,3)-data['pts_gt'],dim=1).mean()))
        gt_pose=data['pose'][i]
        r_loss=(np.trace(pose[:3,:3]@gt_pose[:3,:3].T)-1)/2
        r_loss=np.clip(r_loss,a_min=-1+eps,a_max=1-eps)
        R_error=np.arccos(r_loss)*180/np.pi
        t_error=np.linalg.norm(pose[:3,3]-gt_pose[:3,3])
        print('Predict Pose Error: R_error: {} t_error:{}'.format(R_error,t_error))
        # import pdb;pdb.set_trace()
        track=data['track'][i]
        vis=data['vis'][i]

        vis_x=torch.cat([torch.ones(7).to(torch.bool),vis[:,None].repeat(1,3).reshape(-1)])
        vis_x_cov=(vis_x.to(torch.float32)[:,None]@vis_x.to(torch.float32)[:,None].T).to(torch.bool)

        # ekf_update(x[vis_x],x_cov[vis_x_cov].reshape(vis_x.sum(),-1),pose,track[vis],data['K'])
        # import pdb;pdb.set_trace()
        x[vis_x],_x_cov=ekf_update(x[vis_x],x_cov[vis_x_cov].reshape(vis_x.sum(),-1),pose,track[vis],data['K'])
        x_cov[vis_x_cov]=_x_cov.reshape(-1)
        
        print('Update Points Error :{}'.format(torch.norm(x[7:].reshape(-1,3)-data['pts_gt'],dim=1).mean()))
        pose=torch.eye(4)
        pose[:3,:3]=quat2rotmatrix(x[3:7])
        pose[:3,3]=x[:3]
        gt_pose=data['pose'][i]
        r_loss=(np.trace(pose[:3,:3]@gt_pose[:3,:3].T)-1)/2
        r_loss=np.clip(r_loss,a_min=-1+eps,a_max=1-eps)
        R_error=np.arccos(r_loss)*180/np.pi
        t_error=np.linalg.norm(pose[:3,3]-gt_pose[:3,3])
        print('Update Pose Error: R_error: {} t_error:{}'.format(R_error,t_error))

    print((time.time()-start)/100)
    # import pdb;pdb.set_trace()
        