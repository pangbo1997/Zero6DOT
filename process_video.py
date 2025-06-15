import os
import numpy as np
import random
import cv2
import data_utils
import glob
from tqdm import tqdm


def process_video(video_path,hw=512):

    os.makedirs(os.path.join(video_path,'color_crop_full'),exist_ok=True)
    os.makedirs(os.path.join(video_path,'K_crop_full'),exist_ok=True)

    fx,fy,cx,cy=np.loadtxt(os.path.join(video_path,'intri.txt'))
    K=np.array([[fx,0,cx],
                [0,fy,cy],
                [0,0,1]])*3
    K[2,2]=1
    try:
        for i,img_path in (enumerate(sorted(glob.glob(os.path.join(video_path,'images/*.png'))))):
            image=cv2.imread(img_path.replace('images','color_full'))
            mask=cv2.imread(img_path)
            mask=cv2.resize(cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY).astype(np.bool_).astype(np.uint8),(image.shape[1],image.shape[0])).astype(np.bool_)
            image=image*mask[:,:,None].repeat(3,axis=2)

            reproj_box3d = np.loadtxt(os.path.join(video_path,'reproj_box', '{}.txt'.format(i))).astype(int)
            h,w,_=image.shape
            x0, y0 = reproj_box3d.min(0)
            x1, y1 = reproj_box3d.max(0)

            box = np.array([x0, y0, x1, y1])
            resize_shape = np.array([y1 - y0, x1 - x0])
            K_crop, K_crop_homo = data_utils.get_K_crop_resize(box, K, resize_shape)
            image_crop, trans1 = data_utils.get_image_crop_resize(image, box, resize_shape)

            box_new = np.array([0, 0, x1-x0, y1-y0])
            resize_shape = np.array([hw, hw])
            K_crop, K_crop_homo = data_utils.get_K_crop_resize(box_new, K_crop, resize_shape)
            image_crop, trans2 = data_utils.get_image_crop_resize(image_crop, box_new, resize_shape)


            cv2.imwrite(os.path.join(video_path,'color_crop_full/{:06d}.png'.format(i)),image_crop)
            np.savetxt(os.path.join(video_path,'K_crop_full/{}.txt'.format(i)),np.array([K_crop[0][0],K_crop[1][1],K_crop[0][2],K_crop[1][2]]))
    except:
        print(' ')

    os.system(f'rm -rf {video_path}/rgb_crop_full.mp4')
    os.system(f'ffmpeg -i {video_path}/color_crop_full/%6d.png {video_path}/rgb_crop_full.mp4')




