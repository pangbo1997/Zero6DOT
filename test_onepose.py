import os
import numpy as np
import random
import cv2
import time
data_path='/home/pb/Object-SLAM_onepose/test_data/'
dirs=os.listdir(data_path)
test_obj=open('test_set.txt','r').read().split('\n')
cnt=0


from process_video import process_video

for i,object_name in enumerate(dirs):
    for object_no in os.listdir(os.path.join(data_path,object_name)):
        if os.path.exists(f'results/reconstructions_{object_name}_{object_no}/poses/0.txt'): continue
        if not '{} {}'.format(object_name,object_no) in test_obj: continue
        if object_no=='box3d_corners.txt': continue
        if object_no=='.DS_Store': continue

        full_path=os.path.join(data_path,object_name,object_no)
        print('Processing: {}'.format(full_path))
        if not os.path.exists('{}/color_full/'.format(full_path)):
            os.makedirs('{}/color_full/'.format(full_path),exist_ok=True)
            os.system('ffmpeg -r 20 -i {}/Frames.m4v  {}/color_full/%06d.png'.format(full_path,full_path))
            os.system('ln -sf {}/poses_ba {}/poses'.format(full_path,full_path))

        fx,fy,cx,cy=[float(i.split(':')[-1]) for i in open('{}/intrinsics.txt'.format(full_path)).read().split('\n')]
        intri=np.array([fx,fy,cx,cy])/3
        np.savetxt('{}/intri.txt'.format(full_path),intri)

        print('Preprocess Video')
        # process_video(full_path)
        print('Done')
        os.makedirs(f'results/reconstructions_{object_name}_{object_no}',exist_ok=True)

        os.system('rm -rf {}'.format(full_path+'/pred_results'))

        print('Tracking Begin')
        t=time.time()
        os.system(f'python run.py --imagedir={full_path}/color_crop_full --calib_path={full_path}/K_crop_full --data_base_dir={full_path} --object_no={object_name}_{object_no} --disable_vis --video_path={full_path}/rgb_crop_full.mp4')
        print('Done')
        print('Process time:{}'.format(time.time()-t))




