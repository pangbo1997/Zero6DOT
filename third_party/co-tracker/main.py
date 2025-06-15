# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import cv2
import torch
import argparse
import numpy as np

from PIL import Image
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor
import torchvision
import torch.nn.functional as F

import superpoint_pytorch
DEFAULT_DEVICE = ('cuda' if torch.cuda.is_available() else
                  'mps' if torch.backends.mps.is_available() else
                  'cpu')

def convert_select_points_to_query_points(frame, points):
  """Convert select points to query points.

  Args:
    points: [num_points, 2], [t, y, x]
  Returns:
    query_points: [num_points, 3], [t, y, x]
  """
  points = np.stack(points)
  query_points = np.zeros(shape=(points.shape[0], 3), dtype=np.float32)
  query_points[:, 0] = frame
  query_points[:, 1] = points[:, 0]
  query_points[:, 2] = points[:, 1]
  return query_points
from notebooks.utils import plot_imgs
def meshgrid2d(B, Y, X, stack=False, norm=False, device="cuda"):
    # returns a meshgrid sized B x Y x X

    grid_y = torch.linspace(0.0, Y - 1, Y, device=torch.device(device))
    grid_y = torch.reshape(grid_y, [1, Y, 1])
    grid_y = grid_y.repeat(B, 1, X)

    grid_x = torch.linspace(0.0, X - 1, X, device=torch.device(device))
    grid_x = torch.reshape(grid_x, [1, 1, X])
    grid_x = grid_x.repeat(B, Y, 1)

    if stack:
        # note we stack in xy order
        # (see https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample)
        grid = torch.stack([grid_x, grid_y], dim=-1)
        return grid
    else:
        return grid_y, grid_x
    
def get_points_on_a_grid(grid_size, interp_shape, grid_center=(0, 0), device="cuda"):
    if grid_size == 1:
        return torch.tensor([interp_shape[1] / 2, interp_shape[0] / 2], device=device)[
            None, None
        ]

    grid_y, grid_x = meshgrid2d(
        1, grid_size, grid_size, stack=False, norm=False, device=device
    )
    step = interp_shape[1] // 64
    if grid_center[0] != 0 or grid_center[1] != 0:
        grid_y = grid_y - grid_size / 2.0
        grid_x = grid_x - grid_size / 2.0
    grid_y = step + grid_y.reshape(1, -1) / float(grid_size - 1) * (
        interp_shape[0] - step * 2
    )
    grid_x = step + grid_x.reshape(1, -1) / float(grid_size - 1) * (
        interp_shape[1] - step * 2
    )

    grid_y = grid_y + grid_center[0]
    grid_x = grid_x + grid_center[1]
    xy = torch.stack([grid_x, grid_y], dim=-1).to(device)
    return xy

def get_mask_grid_points(video,frame_idx=0,grid_size=20):
   

    grid_pts = get_points_on_a_grid(grid_size, (video.shape[1],video.shape[2]))
    mask=torch.tensor(video)
    segm_mask=torchvision.transforms.functional.rgb_to_grayscale(mask[frame_idx:frame_idx+1].permute(0, 3, 1, 2).float())
    
    num_labels,labels,stats,centroids=cv2.connectedComponentsWithStats(segm_mask[0][0].numpy().astype(np.uint8))
    max_num=-1
    for label in np.unique(labels):
        mask=(labels==label)
        masksum=mask.sum()
        if mask.sum()>max_num and label!=0:
            segm_mask=mask
            max_num=masksum
    # import pdb;pdb.set_trace()      

    # kernel_2 = np.ones((5, 5), dtype=np.uint8)
    kernel_2 = np.ones((9, 9), dtype=np.uint8)
    segm_mask = cv2.erode(segm_mask.astype(np.uint8)*255, kernel_2, iterations=1).astype(np.bool_)

    segm_mask=torch.tensor(segm_mask[None,None])

    point_mask = segm_mask[0, 0][
        (grid_pts[0, :, 1]).round().long().cpu(),
        (grid_pts[0, :, 0]).round().long().cpu(),
    ].bool()
    grid_pts = grid_pts[:, point_mask]
    select_points=grid_pts.detach().cpu().numpy()[0]
    # h1,w1=video.shape[1],video.shape[2]
    # select_points=(select_points-np.array([w1//2,h1//2]))*0.8+np.array([w1//2,h1//2])
    # import pdb;pdb.set_trace()
    return select_points

import matplotlib.pyplot as plt
import random
import shutil
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        default="./assets/rgb_mask.mp4",
        help="path to a video",
    )
    parser.add_argument(
        "--seg_path",
        default="./assets/rgb_mask.mp4",
        help="path to a video",
    )
    parser.add_argument(
        "--mask_path",
        default="./assets/apple_mask.png",
        help="path to a segmentation mask",
    )

    parser.add_argument(
        "--checkpoint",
        default="./model_cotracker_050000.pth",
        help="cotracker model",
    )

    # parser.add_argument(
    #     "--checkpoint",
    #     default="./cotracker_stride_4_wind_8.pth",
    #     help="cotracker model",
    # )

    parser.add_argument("--grid_size", type=int, default=0, help="Regular grid size")
    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=0,
        help="Compute dense and grid tracks starting from this frame ",
    )

    parser.add_argument(
        "--backward_tracking",
        action="store_true",
        help="Compute tracks in both directions, not only forward",
    )

    parser.add_argument(
        "--low",
        type=int,
        default=400
    )

    parser.add_argument(
        "--high",
        type=int,
        default=500
    )

    parser.add_argument(
        "--data_base_dir",
        type=str,
    )
    args = parser.parse_args()

    video = read_video_from_path(args.video_path)
    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
    mask = read_video_from_path(args.seg_path)


    model = CoTrackerPredictor(checkpoint=args.checkpoint)
    model = model.to(DEFAULT_DEVICE)
    video = video.to(DEFAULT_DEVICE)


    # gap=20
    gap=20
    low=args.low
    high=args.high
    pred_tracks=None

    detection_thresh = 0.0005#0.005
    nms_radius = 5
    sp_th = superpoint_pytorch.SuperPoint(detection_threshold=detection_thresh, nms_radius=nms_radius).eval().cuda()
    # print('Config:', sp_th.conf)
    sp_th.load_state_dict(torch.load('superpoint_v6_from_tf.pth'))
    final_descriptor=[]

    if os.path.exists(f'{args.data_base_dir}/pred_results'):
        shutil.rmtree(f'{args.data_base_dir}/pred_results')
    os.makedirs(f'{args.data_base_dir}/pred_results',exist_ok=True)
    
    with torch.no_grad():

        for i in range(0,video.shape[1], gap):
            
            
            gray_img=torchvision.transforms.functional.rgb_to_grayscale(video[0,i])[0].detach().cpu().numpy().astype(np.uint8)
            kernel_2 = np.ones((5, 5), dtype=np.uint8)
            gray_img = cv2.erode(gray_img.astype(np.uint8), kernel_2, iterations=2)
            gray_img=gray_img/255.


            if i!=0:
                last_pred_tracks=torch.cat((torch.zeros(last_vis.sum(),1),last_vis_track[last_vis]),dim=1)
                flag_low=False
                flag_high=False
                new_queries=None
                last_shape=None
                for _ in range(100):
                    sp_th.conf.detection_threshold=detection_thresh
                    with torch.no_grad(): pred_th = sp_th({'image': torch.from_numpy(gray_img[None,None]).float().cuda()})
                    points_th = pred_th['keypoints'][0].detach().cpu().numpy()
                    if new_queries is not None: last_shape=new_queries.shape[1]
                    select_idx=gray_img[points_th.astype(np.uint32)[:,1],points_th.astype(np.uint32)[:,0]]!=0
                    new_queries=points_th[select_idx]
                    if len(new_queries)==0:
                        if last_pred_tracks.shape[0]>low:
                            new_queries=torch.zeros(1,0,3).cuda() 
                            break
                        else:
                            for _ in range(100):
                                detection_thresh/=2
                                sp_th.conf.detection_threshold=detection_thresh
                                with torch.no_grad(): pred_th = sp_th({'image': torch.from_numpy(gray_img[None,None]).float().cuda()})
                                points_th = pred_th['keypoints'][0].detach().cpu().numpy()
                                select_idx=gray_img[points_th.astype(np.uint32)[:,1],points_th.astype(np.uint32)[:,0]]!=0
                                new_queries=points_th[select_idx]

                    new_queries=torch.tensor(convert_select_points_to_query_points(0,new_queries))[None].cuda()
                    
        
                    if len(last_pred_tracks)!=0:
                        dis=torch.cdist(new_queries[0,:,1:],last_pred_tracks[:,1:].cuda())
                        select_idx2=dis.min(dim=1)[0]>5
                        new_queries=new_queries[:,select_idx2]

                    if last_shape is not None and new_queries.shape[1]==last_shape: break
                    if last_pred_tracks.shape[0]+new_queries.shape[1]>high:
                        detection_thresh*=2
                        if flag_low: break
                        flag_high=True
                        flag_low=False
                    elif last_pred_tracks.shape[0]+new_queries.shape[1]<low:
                        detection_thresh/=2
                        if flag_high: break
                        flag_low=True
                        flag_high=False
                    else:
                        break

                new_queries[:,:,0]+=1

                if len(last_pred_tracks)+new_queries.shape[1]>800:
                    new_queries=new_queries[:,:(800-len(last_pred_tracks))]
                queries=torch.cat((last_pred_tracks[None].cuda(),new_queries),dim=1)

                local_video=video[:,i-1:i+ gap] 
            else:
                new_queries=None
                flag_low=False
                flag_high=False
                last_shape=None
                for _ in range(100):                    
                    sp_th.conf.detection_threshold=detection_thresh
                    with torch.no_grad(): pred_th = sp_th({'image': torch.from_numpy(gray_img[None,None]).float().cuda()})
                    points_th = pred_th['keypoints'][0].detach().cpu().numpy()
                    if new_queries is not None: last_shape=new_queries.shape[0]
                    select_idx=gray_img[points_th.astype(np.uint32)[:,1],points_th.astype(np.uint32)[:,0]]!=0
                    new_queries=points_th[select_idx]
                    if last_shape is not None:
                        if new_queries.shape[0]==last_shape: break

                    # import pdb;pdb.set_trace()
                    if new_queries.shape[0]>high:
                        detection_thresh*=2
                        if flag_low: break
                        flag_high=True
                        flag_low=False
                    elif new_queries.shape[0]<low:
                        detection_thresh/=2                        
                        if flag_high: break
                        flag_low=True
                        flag_high=False

                    else:
                        break

                new_queries=torch.tensor(convert_select_points_to_query_points(0,new_queries))[None].cuda()
                queries=new_queries
                local_video=video[:,:gap]   

            # print(i,queries.shape)

            pred_tracks, pred_visibility, ffeats= model(
                local_video,
                grid_size=args.grid_size,
                grid_query_frame=args.grid_query_frame,
                backward_tracking=args.backward_tracking,
                queries=queries
            )

            if i!=0:

                pred_tracks=pred_tracks[:,1:,].detach().cpu()
                pred_visibility=pred_visibility[:,1:].detach().cpu()

                features=sp_th.backbone(torch.from_numpy(gray_img[None,None]).float().cuda())
                descriptors_dense = torch.nn.functional.normalize(sp_th.descriptor(features), p=2, dim=1)
                descriptors=superpoint_pytorch.sample_descriptors(pred_tracks[0,0].cuda(),descriptors_dense,sp_th.stride)[0].transpose(1,0)

                old_track=torch.zeros(1,pred_tracks.shape[1],final_pred_tracks.shape[2],2)
                old_vis=torch.zeros(1,pred_tracks.shape[1],final_pred_vis.shape[2],dtype=torch.bool)
                old_track[:,:,last_vis,:]=pred_tracks[:,:,:last_query_length,:]
                old_vis[:,:,last_vis]=pred_visibility[:,:,:last_query_length]

                
                final_pred_tracks=torch.cat((final_pred_tracks,old_track),dim=1)
                final_pred_vis=torch.cat((final_pred_vis,old_vis),dim=1)
                


                new_track=torch.zeros(1,final_pred_tracks.shape[1],pred_tracks.shape[2]-last_query_length,2)#.cuda()
                new_track[:,-pred_tracks.shape[1]:,:,:]=pred_tracks[:,:,last_query_length:,:]
                new_vis=torch.zeros(1,final_pred_tracks.shape[1],pred_tracks.shape[2]-last_query_length,dtype=torch.bool)#.cuda()
                new_vis[:,-pred_visibility.shape[1]:,:]=pred_visibility[:,:,last_query_length:]
                final_pred_tracks=torch.cat((final_pred_tracks,new_track),dim=2)
                final_pred_vis=torch.cat((final_pred_vis,new_vis),dim=2)


                choose_idx=torch.cat((last_vis,torch.ones(new_queries.shape[1]).to(torch.bool)))
                last_vis_track=final_pred_tracks[0,-1]
 
                # last_vis=final_pred_vis[0,-gap:].sum(dim=0)>(0.75*gap)
                last_vis=final_pred_vis[0,-1]
                

                frame_idx=torch.cat((frame_idx,torch.zeros(new_queries.shape[1])+i))
                last_query_length=last_vis.sum()

            else:
                pred_tracks=pred_tracks.detach().cpu()
                pred_visibility=pred_visibility.detach().cpu()   

                features=sp_th.backbone(torch.from_numpy(gray_img[None,None]).float().cuda())
                descriptors_dense = torch.nn.functional.normalize(sp_th.descriptor(features), p=2, dim=1)
                descriptors=superpoint_pytorch.sample_descriptors(pred_tracks[0,0].cuda(),descriptors_dense,sp_th.stride)[0].transpose(1,0)

                final_pred_tracks=pred_tracks
                final_pred_vis=pred_visibility
                last_vis_track=pred_tracks[0,-1]
                # last_vis=pred_visibility[0,-gap:].sum(dim=0)>(0.75*gap)
                last_vis=pred_visibility[0,-1]
                frame_idx=torch.zeros(new_queries.shape[1])+i
                last_query_length=last_vis.sum()
                choose_idx=torch.ones(new_queries.shape[1]).to(torch.bool)
            
            final_descriptor.append(descriptors)


            torch.save(final_pred_tracks[0][i:i+gap,choose_idx],'{}/pred_results/pred_tracks_{:06d}.dict'.format(args.data_base_dir,i//gap))
            torch.save(final_pred_vis[0][i:i+gap,choose_idx],'{}/pred_results/pred_vis_{:06d}.dict'.format(args.data_base_dir,i//gap))
            torch.save(final_pred_vis[0][i:i+gap,choose_idx],'{}/pred_results/pred_vis_{:06d}.dict'.format(args.data_base_dir,i//gap))
            torch.save(ffeats,'{}/pred_results/ffeats_{:06d}.dict'.format(args.data_base_dir,i//gap))
            torch.save(torch.arange(frame_idx.shape[0])[choose_idx],'{}/pred_results/vertex_ids_{:06d}.dict'.format(args.data_base_dir,i//gap))
            torch.save(frame_idx[choose_idx],'{}/pred_results/frame_idx_{:06d}.dict'.format(args.data_base_dir,i//gap))
            torch.save(descriptors,'{}/pred_results/descriptors_{:06d}.dict'.format(args.data_base_dir,i//gap))
            torch.save(torch.tensor(False),'{}/pred_results/need_reinit_{:06d}.dict'.format(args.data_base_dir,i//gap))




