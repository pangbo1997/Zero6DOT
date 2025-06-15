import cv2
import torch

class MotionFilter:
    """ This class is used to filter incoming frames and extract features """
    def __init__(self, video, thresh=2.5, device="cuda:0"):
        
        self.video = video
        self.thresh = thresh
        self.device = device

        self.image=None
        self.count = 0

    @torch.cuda.amp.autocast(enabled=True)
    @torch.no_grad()
    def track(self, tstamp, image, depth=None, intrinsics=None):
        """ main update operation - run on every frame in video """

        self.video.append(tstamp, image, None, None, depth, intrinsics)







