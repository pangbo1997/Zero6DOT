import cv2
import numpy as np 
import open3d as o3d
from utils import *

def get_uvcoord(point_cloud,depth_cam_matrix,width,height):
    fx,fy = depth_cam_matrix[0,0],depth_cam_matrix[1,1]
    cx,cy = depth_cam_matrix[0,2],depth_cam_matrix[1,2]
    x=point_cloud[:,0]
    y=point_cloud[:,1]
    z=point_cloud[:,2]
    w_corrd=((x*fx/z)+cx).astype(np.int64)
    h_corrd=((y*fy/z)+cy).astype(np.int64)



    return h_corrd,w_corrd


class Visualizer:
    def __init__(self,args,frame_id_offset) -> None:

        os.makedirs(f'results/reconstructions_{args.object_no}/pcd_file',exist_ok=True)
        os.makedirs(f'results/reconstructions_{args.object_no}/images',exist_ok=True)
        os.makedirs(f'results/reconstructions_{args.object_no}/poses',exist_ok=True)

        os.system(f'rm -rf results/reconstructions_{args.object_no}/pcd_file/*')
        os.system(f'rm -rf results/reconstructions_{args.object_no}/images/*')
        os.system(f'rm -rf results/reconstructions_{args.object_no}/poses/*')
            
        self.args=args
        self.frame_id_offset=frame_id_offset

    def project(self,xyz, K, RT):
        """
        xyz: [N, 3]
        K: [3, 3]
        RT: [3, 4]
        """
        xyz = np.dot(xyz, RT[:3, :3].T) + RT[:3, 3:].T
        xyz = np.dot(xyz, K.T)
        xy = xyz[:, :2] / xyz[:, 2:]
        return xy 
    def draw_box(self,box3d_proj, image=None, plt=None, color=None, line_width=7,color_id=0):
        import cv2
        lines = [
                [1, 2],
                [1, 3],
                [1, 5],
                [2, 6],
                [2, 4],
                [4, 3],
                [4, 8],
                [8, 6],
                [8, 7],
                [5, 7],
                [3, 7],
                [5, 6]
        ]
        #import pdb;pdb.set_trace()
        draw_box3d=box3d_proj[[6,2,5,1,7,3,4,0],:]
        # orange, blue, green, purple red, pink, caffe
        color_list = [(47, 92, 247), (68, 27, 203), (221, 168, 81), (62, 129, 27), (129, 51, 111), (27, 177, 255), (56, 67, 115), (47, 92, 247), (68, 27, 203), (221, 168, 81), (62, 129, 27), (27, 177, 255)]
        color=color_list[color_id]
        for id, line in enumerate(lines):
            pt1 = draw_box3d[line[0] - 1, :]
            pt2 = draw_box3d[line[1] - 1, :]

            if image is not None:
                pt1 = (int(pt1[0]), int(pt1[1]))
                pt2 = (int(pt2[0]), int(pt2[1]))
                # import pdb;pdb.set_trace()
                cv2.line(image, pt1, pt2, color, line_width)

            if plt is not None:
                plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]])

    def visualize(self,graph,fragment,length):
        
        pts=graph.get_points()
        length=graph.last_fragment.track.shape[0]
        for i,frame_id in enumerate(range(graph.frame_id_offset-length,graph.frame_id_offset)):
            if graph.need_reinit:
                # import pdb;pdb.set_trace()
                img=fragment.images[i].permute(1,2,0).cpu().numpy().copy()
                cv2.imwrite(f'results/reconstructions_{self.args.object_no}/images/{frame_id-self.frame_id_offset}.png',img)
                cv2.imwrite(f'results/reconstructions_{self.args.object_no}/images/{frame_id-self.frame_id_offset}_0.png',img)
            else:
                img=graph.last_fragment.images[i]
                pose=graph.optimizer.vertex(frame_id).estimate()
                t=pose.translation()
                R=pose.rotation().matrix()
                RT=np.zeros((3,4))
                RT[:3,:3]=R
                RT[:3,3]=t
                np.savetxt(f'results/reconstructions_{self.args.object_no}/poses/{frame_id-self.frame_id_offset}.txt',RT)
                fx,fy,cx,cy=fragment.intrinsics[i].cpu().numpy()
                # import pdb;pdb.set_trace()
                pts, ind = pts.remove_radius_outlier(nb_points=16, radius=0.1)
                box_3d=np.array(pts.get_axis_aligned_bounding_box().get_box_points())[[0,1,6,3,2,7,4,5]]
                K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
                pts2d=self.project(box_3d, K, RT)
                img_draw=img.permute(1,2,0).detach().cpu().numpy().copy()
                self.draw_box(pts2d,img_draw , line_width=3)
                cv2.imwrite(f'results/reconstructions_{self.args.object_no}/images/{frame_id-self.frame_id_offset}.png',img_draw)
                o3d.io.write_point_cloud(f'results/reconstructions_{self.args.object_no}/pcd_file/{frame_id-self.frame_id_offset}.ply',pts)
                pts_array=(np.array(pts.points))@R.T+t
                colors=np.array(pts.colors)
                _,mask_map=point_cloud_to_depth(pts_array,K,img.shape[2],img.shape[1])
                img1=img.permute(1,2,0).cpu().numpy().copy()
                pos1=np.array(mask_map.nonzero()).T
                
    
                for i in range(pos1.shape[0]):
                    img1 = cv2.circle(img1, (pos1[i][1].astype(int),pos1[i][0].astype(int)), 3,(200.,200.,200.), -1)
                cv2.imwrite(f'results/reconstructions_{self.args.object_no}/images/{frame_id-self.frame_id_offset}_0.png',img1)
    
    def save_final_point_cloud(self,graph):
        
        pts=graph.get_points()
        o3d.io.write_point_cloud(f'results/reconstructions_{self.args.object_no}/pcd_file/final.ply',pts)
               
    def visualize_gt(self,graph,fragment,length):
        last_pose=None
        pts=graph.get_points()
        length=graph.last_fragment.track.shape[0]
        for i,frame_id in enumerate(range(graph.frame_id_offset-length,graph.frame_id_offset)):
            if graph.need_reinit:
                # import pdb;pdb.set_trace()
                img=fragment.images[i].permute(1,2,0).cpu().numpy().copy()
                cv2.imwrite(f'results/reconstructions_{self.args.object_no}/images/{frame_id-self.frame_id_offset}.png',img)
                cv2.imwrite(f'results/reconstructions_{self.args.object_no}/images/{frame_id-self.frame_id_offset}_0.png',img)
            else:
                img=graph.last_fragment.images[i]

                if graph.optimizer.vertex(frame_id) is None:
                    pose=last_pose
                else:
                    pose=graph.optimizer.vertex(frame_id).estimate()
                    last_pose=pose
                t=pose.translation()
                R=pose.rotation().matrix()
                RT=np.zeros((3,4))
                RT[:3,:3]=R
                RT[:3,3]=t
                np.savetxt(f'results/reconstructions_{self.args.object_no}/poses/{frame_id-self.frame_id_offset}.txt',RT)
                fx,fy,cx,cy=fragment.intrinsics[i].cpu().numpy()#graph.K[0,0],graph.K[1,1],graph.K[0,2],graph.K[1,2]
                # # import pdb;pdb.set_trace()
                # # pts, ind = pts.remove_radius_outlier(nb_points=16, radius=0.1)
                # # box_3d=np.array(pts.get_axis_aligned_bounding_box().get_box_points())[[0,1,6,3,2,7,4,5]]
                box_3d=np.loadtxt(os.path.join('/',*self.args.data_base_dir.split('/')[:-1],'box3d_corners.txt'))


                K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
                pts2d=self.project(box_3d, K, RT)
                img_draw=img.permute(1,2,0).detach().cpu().numpy().copy()
                self.draw_box(pts2d,img_draw , line_width=5,color_id=2)
                gt_RT=np.loadtxt(self.args.data_base_dir+f'/poses/{frame_id-self.frame_id_offset}.txt')
                self.draw_box(self.project(box_3d, K, gt_RT),img_draw , line_width=5,color_id=3)
                cv2.imwrite(f'results/reconstructions_{self.args.object_no}/images/{frame_id-self.frame_id_offset}.png',img_draw)
                # o3d.io.write_point_cloud(f'results/reconstructions_{self.args.object_no}/pcd_file/{frame_id-self.frame_id_offset}.ply',pts)
                
                
                # pts_3d=[]
                # pts_2d=[]
                # for j in range(len(fragment.vertex_ids)):
                #     if graph.optimizer.vertex(fragment.vertex_ids[j]) is not None:
                #         pts_3d.append(graph.optimizer.vertex(fragment.vertex_ids[j]).estimate())
                #         pts_2d.append(fragment.track[i][j])
                # pts_3d=np.stack(pts_3d)
                # pts_2d=np.stack(pts_2d)
                # pts_array=(pts_3d)@R.T+t
                # pos1=np.stack(get_uvcoord(pts_array,K,img.shape[2],img.shape[1])).transpose(1,0)
                
                # img1=img.permute(1,2,0).cpu().numpy().copy()
                # # import pdb;pdb.set_trace()
                # for j in range(pos1.shape[0]):
                #     img1 = cv2.circle(img1, (pos1[j][1].astype(int),pos1[j][0].astype(int)), 3,(200.,200.,200.), -1)

                # for j in range(pts_2d.shape[0]):
                #     # import pdb;pdb.set_trace()
                #     img1 = cv2.circle(img1, (int(pts_2d[j][0].item()),int(pts_2d[j][1].item())), 3,(100.,200.,100.), -1)

                    
                # cv2.imwrite(f'results/reconstructions_{self.args.object_no}/images/{frame_id-self.frame_id_offset}_0.png',img1)

    def visualize_ycb(self,graph,fragment,length):
        
        pts=graph.get_points()
        length=graph.last_fragment.track.shape[0]
        for i,frame_id in enumerate(range(graph.frame_id_offset-length,graph.frame_id_offset)):
            if graph.need_reinit:
                # import pdb;pdb.set_trace()
                img=fragment.images[i].permute(1,2,0).cpu().numpy().copy()
                cv2.imwrite(f'results/reconstructions_{self.args.object_no}/images/{frame_id-self.frame_id_offset}.png',img)
                cv2.imwrite(f'results/reconstructions_{self.args.object_no}/images/{frame_id-self.frame_id_offset}_0.png',img)
            else:
                img=graph.last_fragment.images[i]
                pose=graph.optimizer.vertex(frame_id).estimate()
                t=pose.translation()
                R=pose.rotation().matrix()
                RT=np.zeros((3,4))
                RT[:3,:3]=R
                RT[:3,3]=t
                np.savetxt(f'results/reconstructions_{self.args.object_no}/poses/{frame_id-self.frame_id_offset}.txt',RT)
                fx,fy,cx,cy=fragment.intrinsics[i].cpu().numpy()#graph.K[0,0],graph.K[1,1],graph.K[0,2],graph.K[1,2]
                K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
                # import pdb;pdb.set_trace()
                pts, ind = pts.remove_radius_outlier(nb_points=16, radius=0.1)
                # box_3d=np.array(pts.get_axis_aligned_bounding_box().get_box_points())[[0,1,6,3,2,7,4,5]]

                box_3d=np.loadtxt(os.path.join(self.args.data_base_dir,'box3d_corners.txt'))


                K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
                pts2d=self.project(box_3d, K, RT)
                img_draw=img.permute(1,2,0).detach().cpu().numpy().copy()
                self.draw_box(pts2d,img_draw , line_width=4, color_id=2)
                if 'init' in self.args.imagedir:
                    gt_RT=np.loadtxt(self.args.data_base_dir+f'/poses_init/{frame_id-self.frame_id_offset}.txt')
                else:
                    gt_RT=np.loadtxt(self.args.data_base_dir+f'/poses/{frame_id-self.frame_id_offset}.txt')
                self.draw_box(self.project(box_3d, K, gt_RT),img_draw , line_width=4,color_id=3)
                cv2.imwrite(f'results/reconstructions_{self.args.object_no}/images/{frame_id-self.frame_id_offset}.png',img_draw)
                o3d.io.write_point_cloud(f'results/reconstructions_{self.args.object_no}/pcd_file/{frame_id-self.frame_id_offset}.ply',pts)
                pts_array=(np.array(pts.points))@R.T+t
                colors=np.array(pts.colors)
                _,mask_map=point_cloud_to_depth(pts_array,K,img.shape[2],img.shape[1])
                img1=img.permute(1,2,0).cpu().numpy().copy()
                pos1=np.array(mask_map.nonzero()).T
                
    
                for i in range(pos1.shape[0]):
                    img1 = cv2.circle(img1, (pos1[i][1].astype(int),pos1[i][0].astype(int)), 3,(200.,200.,200.), -1)
                cv2.imwrite(f'results/reconstructions_{self.args.object_no}/images/{frame_id-self.frame_id_offset}_0.png',img1)
