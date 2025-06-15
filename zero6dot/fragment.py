from typing import Any
import numpy as np
import torch


from torch.multiprocessing import Process, Queue, Lock, Value




class Fragment:
    def __init__(self,fragment_id,need_reinit,track,vis,ffeats,descriptors,vertex_ids,images,intrinsics):
                
        self.fragment_id=fragment_id
        self.need_reinit=need_reinit
        self.images=images
        self.intrinsics=intrinsics
        self.ffeats=ffeats
        if not need_reinit:
            self.track=track
            self.visible=vis
            self.descriptors=descriptors
            self.vertex_ids=vertex_ids.detach().cpu().numpy()




class Frame:
    def __init__(self,optimizer,vertex_id,pose,descriptors,pts_vertex_ids,fixed=False):
        self.optimizer=optimizer
        self.vertex_id=vertex_id
        self.pose=pose
        self.fixed=fixed
        self.age=0
        self.descirptors=descriptors
        self.pts_vertex_ids=pts_vertex_ids

class Point:
    def __init__(self,optimizer,vertex_id,pts_xyz,color,fixed=False):
        self.optimizer=optimizer
        self.vertex_id=vertex_id
        self.pts_xyz=pts_xyz
        self.color=color
        self.fixed=fixed
        self.age=0




    