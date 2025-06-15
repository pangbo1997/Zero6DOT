import torch
import numpy as np

from video import Video
from motion_filter import MotionFilter
from graph import Graph

from collections import OrderedDict

from tqdm import tqdm
from fragment import Fragment
from visualizer import Visualizer
import time

class Zero6DOT:
    def __init__(self, args):
        super(Zero6DOT, self).__init__()

        self.args = args
        self.disable_vis = args.disable_vis
        self.video = Video(args.image_size, args.buffer, args=args, stereo=args.stereo)
        self.filterx = MotionFilter(self.video, thresh=args.filter_thresh)
        self.graph= Graph(args,self.video)
        self.visualizer=Visualizer(args,self.graph.frame_id_offset)

    def run(self, image_stream,fragment_stream):
        """ main thread - update map """

        for (tstamp, image, intrinsics) in (image_stream):
            with torch.no_grad():
                self.filterx.track(tstamp, image, None, intrinsics)
    
        for t, need_reinit,track,vis,ffeats,descriptors,vertex_ids in tqdm(fragment_stream):
            if need_reinit:
                length=self.graph.gap
            else:
                length=track.shape[0]

            start=self.graph.frame_id_offset-self.graph.fist_frame_id
            end=start+length
            fragment=Fragment(t, need_reinit,track,vis,ffeats,descriptors,vertex_ids,self.video.images[start:end],self.video.intrinsics[start:end])
            self.graph.collect_img_global_feature(fragment)
            start=time.time()
            self.graph.frontend(fragment)
            start=time.time()
            self.graph.update_offset(length)
            start=time.time()
            self.visualizer.visualize_gt(self.graph,fragment,length)
            self.graph.backend()

            




