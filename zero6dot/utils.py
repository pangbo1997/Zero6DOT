import torch

import numpy as np

import matplotlib.pyplot as plt
import os
os.environ["CUDA_LAUNCH_BLOCKING"]="1"
def plot_depth(depth_buf,i):
    depth=depth_buf[i].cpu().numpy()
    depth[depth==1]=0
    plt.imshow(depth)
    plt.show()

import open3d as o3d
import matplotlib.colors as mcolors
import random
import copy
import numpy as np
import time
import torchvision
def get_color_map(N=100):
    CSS_colors = mcolors.CSS4_COLORS
    color_set = list(set([mcolors.to_rgb(v) for v in CSS_colors.values()]))
    saved_map = copy.deepcopy(color_set)
    while N >= len(saved_map):
        random.shuffle(color_set)
        saved_map += color_set
    return saved_map[:N]
import open3d as o3d


import g2o
import cv2

def depth_to_pointcloud(depth, K):
    
    vs, us = depth.nonzero()
    zs = depth[vs, us]
    xs = (us - K[0, 2]) * zs / K[0, 0]
    ys = (vs - K[1, 2]) * zs / K[1, 1]
    pts = np.stack([xs, ys, zs], axis=1)
    return pts

def point_cloud_to_depth(point_cloud,depth_cam_matrix,width,height):
    fx,fy = depth_cam_matrix[0,0],depth_cam_matrix[1,1]
    cx,cy = depth_cam_matrix[0,2],depth_cam_matrix[1,2]
    x=point_cloud[:,0]
    y=point_cloud[:,1]
    z=point_cloud[:,2]
    w_corrd=((x*fx/z)+cx).astype(np.int64)
    h_corrd=((y*fy/z)+cy).astype(np.int64)

    depth_map=np.zeros((height,width)).astype(np.int32)
    mask_map=np.zeros((height,width)).astype(np.uint8)



    h_corrd[h_corrd<0]=0
    w_corrd[w_corrd<0]=0
    h_corrd[h_corrd>=height]=height-1
    w_corrd[w_corrd>=width]=width-1

    mask_map[h_corrd,w_corrd]=1
    depth_map[h_corrd,w_corrd]=z
    return depth_map,mask_map


def get_uvcoord(point_cloud,depth_cam_matrix,width,height):
    fx,fy = depth_cam_matrix[0,0],depth_cam_matrix[1,1]
    cx,cy = depth_cam_matrix[0,2],depth_cam_matrix[1,2]
    x=point_cloud[:,0]
    y=point_cloud[:,1]
    z=point_cloud[:,2]
    w_corrd=((x*fx/z)+cx).astype(np.int64)
    h_corrd=((y*fy/z)+cy).astype(np.int64)



    return h_corrd,w_corrd

import matplotlib
def make_matching_figure(
        img0, img1, mkpts0, mkpts1, color,
        kpts0=None, kpts1=None, text=[], dpi=75, path=None):
    # draw image pair
    assert mkpts0.shape[0] == mkpts1.shape[0], f'mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}'
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(img0, cmap='gray')
    axes[1].imshow(img1, cmap='gray')
    for i in range(2):   # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)
    
    if kpts0 is not None:
        assert kpts1 is not None
        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c='w', s=2)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c='w', s=2)

    # draw matches
    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        fig.lines = [matplotlib.lines.Line2D((fkpts0[i, 0], fkpts1[i, 0]),
                                            (fkpts0[i, 1], fkpts1[i, 1]),
                                            transform=fig.transFigure, c=color[i], linewidth=1)
                                        for i in range(len(mkpts0))]
        
        axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color, s=4)
        axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color, s=4)

    # put txts
    txt_color = 'k' if img0[:100, :200].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)

    # save or return figure
    if path:
        plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        return fig
    


            






        
