import sys
sys.path.append('zero6dot')
import glob
from tqdm import tqdm
import numpy as np
import torch
import cv2
import os
import glob 
import time
import argparse
from zero6dot import Zero6DOT


def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)

def image_stream(imagedir, calib_path, stride):
    """ image generator """
    image_list = sorted(os.listdir(imagedir))
    for t, imfile in enumerate(image_list):
        image = cv2.imread(os.path.join(imagedir, imfile))
        calib = np.loadtxt(os.path.join(calib_path,f'{t}.txt'), delimiter=" ")
        fx, fy, cx, cy = calib[:4]
        K = np.eye(3)
        K[0,0] = fx
        K[0,2] = cx
        K[1,1] = fy
        K[1,2] = cy
        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])
        image = torch.as_tensor(image).permute(2, 0, 1)
        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        yield t, image[None], intrinsics



def fragment_stream(fragment_dir,max_iter):
    """ image generator """
    for t,i in enumerate(range(max_iter)):
        file=os.path.join(fragment_dir,'need_reinit_{:06d}.dict'.format(i))
        while not os.path.exists(file):
            time.sleep(0.1)
        need_reinit=torch.load(file)
        track=torch.load(file.replace('need_reinit','pred_tracks'))
        vis=torch.load(file.replace('need_reinit','pred_vis'))
        ffeats=torch.load(file.replace('need_reinit','ffeats'))
        descriptors=torch.load(file.replace('need_reinit','descriptors'))
        vertex_id=torch.load(file.replace('need_reinit','vertex_ids'))
        # import pdb;pdb.set_trace()
        yield t, need_reinit,track,vis,ffeats,descriptors,vertex_id




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagedir", type=str, help="path to image directory")
    parser.add_argument("--calib_path", type=str, help="path to calibration file")
    parser.add_argument("--t0", default=0, type=int, help="starting frame")
    parser.add_argument("--stride", default=3, type=int, help="frame stride")

    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=512)
    parser.add_argument("--image_size", default=[240, 320])
    parser.add_argument("--disable_vis", action="store_true")

    parser.add_argument("--beta", type=float, default=0.3, help="weight for translation / rotation components of flow")
    parser.add_argument("--filter_thresh", type=float, default=2.4, help="how much motion before considering new keyframe")

    parser.add_argument("--reconstruction_path", help="path to saved reconstruction")

    parser.add_argument("--data_base_dir", type=str)
    parser.add_argument("--object_no", type=str)


    parser.add_argument(
        "--video_path",
        default="./assets/rgb_mask.mp4",
        help="path to a video",
    )

    parser.add_argument(
        "--low",
        type=int,
        default=600
    )

    parser.add_argument(
        "--high",
        type=int,
        default=700
    )

    args = parser.parse_args()
    args.stereo = False

    tstamps = []

    args.video_path=os.path.join(os.getcwd(),args.video_path)


    os.system('cd third_party/co-tracker/   && python main.py --low {} --high {} --grid_size 80 --video_path {} --seg_path {} --data_base_dir {} &'.format(args.low,args.high,args.video_path,args.video_path,args.data_base_dir))


    img=cv2.imread(glob.glob(os.path.join(args.imagedir,'*.png'))[0])

    args.image_size = [img.shape[0],img.shape[1]]

    engine = Zero6DOT(args)
    
    args.fragment_dir=os.path.join(args.data_base_dir,'pred_results')
    gap=20
    img_num=len(glob.glob(args.imagedir+'/*.png'))
    max_iter=img_num//gap+(img_num%gap!=0)
    engine.run(image_stream(args.imagedir, args.calib_path, args.stride),fragment_stream(args.fragment_dir,max_iter))



