import os
import numpy as np
import random
import cv2
data_path='/home/pb/Object-SLAM_onepose/test_data/'

import json
test_obj=open('test_set.txt','r').read().split('\n')

def generate_json():
    total_instance=0
    total_result=[]
    obj_list=[]
    dirs=os.listdir(data_path)
    separator = '-' 
    for i,object_name in enumerate(dirs):

        for object_no in os.listdir(os.path.join(data_path,object_name)):
            if not '{} {}'.format(object_name,object_no) in test_obj: continue
            if object_no=='box3d_corners.txt': continue
            if object_no=='.DS_Store': continue
            full_path=os.path.join(data_path,object_name,object_no) 
            R_error_list=[]
            t_error_list=[]

            pred_pose_dir=f'results/reconstructions_{object_name}_{object_no}/poses'
            
            if not os.path.exists(pred_pose_dir):
                print('#################################')
                print(pred_pose_dir)
                print('#################################')
                continue
                # pred_pose_dir=f'results_onepose/reconstructions_{object_name}_{object_no}/poses'

            gt_pose_dir= os.path.join(full_path,'poses')
            # import pdb;pdb.set_trace()
            eps=1e-6
            count=0
            total=0
            for idx,file in enumerate(os.listdir(gt_pose_dir)):
                if (idx)%5!=0: continue
                file=f'{idx}.txt'
                # import pdb;pdb.set_trace()
            # for file in tqdm(os.listdir(pred_pose_dir)):
                if file[-3:]!='txt': continue
                if file not in os.listdir(pred_pose_dir): continue

                
                pred_pose=np.loadtxt(os.path.join(pred_pose_dir,file))
                gt_pose=np.loadtxt(os.path.join(gt_pose_dir,file))
                r_loss=(np.trace(pred_pose[:3,:3]@gt_pose[:3,:3].T)-1)/2
                r_loss=np.clip(r_loss,a_min=-1+eps,a_max=1-eps)
                R_error=np.arccos(r_loss)*180/np.pi
                t_error=np.linalg.norm(pred_pose[:3,3]-gt_pose[:3,3])
                if R_error<5 and t_error<0.05:
                    count+=1
                R_error_list.append(R_error)
                t_error_list.append(t_error)
                total+=1

            if total==0: continue
            print(i,full_path)
            print('5rad 5cm:{}'.format(count/total))
            print('R_error:{}'.format(np.mean(R_error_list)))
            print('t_error:{}'.format(np.mean(t_error_list)))

            total_instance+=1
            total_result.append(count/total)
            obj_list.append(full_path)




    record={}
    total_result=np.array(total_result)

    for i in range(len(obj_list)):
        obj=obj_list[i]
        obj_key=separator.join(obj.split('/')[-2:])
        record[obj_key]=total_result[i]

    record['general_profile']={}
    obj_list=np.array(obj_list)
    for i in range(10):
        idx=(total_result>(i/10)) & (total_result<=((i+1)/10))
        key='{:.1f}-{:.1f}'.format(i/10,(i+1)/10)
        record['general_profile'][key]={}
        record['general_profile'][key]['percentage']=idx.sum()/total_instance
        record['general_profile'][key]['obj_list']=obj_list[idx].tolist()
        record['general_profile'][key]['obj_scores']=total_result[idx].tolist()





    result=[]
    for obj_file in test_obj:
        obj_key='-'.join(obj_file.split(' '))
        tmp=[]
        try:
            tmp.append(record[obj_key])
        except TypeError:
            import pdb;pdb.set_trace()

        result.append(np.max(tmp))
        print(obj_key,result[-1])
    print(np.mean(result))
    record['average']=np.mean(result)

    with open('result_onepose.json','w') as f:
        json.dump(record,f,indent=2)
    return record


record=generate_json()

    