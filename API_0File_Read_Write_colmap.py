import numpy as np
import subprocess

def read_colmap_images_file(images_file_path):
    with open(images_file_path, 'r') as file:
        lines = file.readlines()
    
    camera_poses = []
    for line in lines:
        if line.startswith('#'):
            
            continue
        elements = line.split()
        #IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        if len(elements) == 10:
            image_id = int(elements[0])
            qw, qx, qy, qz = map(float, elements[1:5])
            tx, ty, tz = map(float, elements[5:8])
            camera_id= elements[8]
            image_name= elements[9]
            camera_poses.append((image_id, qw, qx, qy, qz, tx, ty, tz))
         
    
    return camera_poses

# 四元数转化为R矩阵
def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    # Compute rotation matrix from quaternion
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    return R

# R矩阵和t矩阵转换为T矩阵
def camera_pose_to_T_4x4(qw, qx, qy, qz, tx, ty, tz):
    R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
    t = np.array([tx, ty, tz]).reshape((3, 1))
    T_3x4 = np.hstack((R, t))
    T_4x4 = np.vstack((T_3x4, [0, 0, 0, 1]))
    return T_4x4

def images_pose_T4x4(camera_poses,txt_name):
    T_4x4_list=[]
    for i, pose in enumerate(camera_poses):
        image_id, qw, qx, qy, qz, tx, ty, tz = pose
        T_Rt4x4 = camera_pose_to_T_4x4(qw, qx, qy, qz, tx, ty, tz)
        
        T_4x4_list_i=[]
        T_4x4_list_i.append(image_id)
        for T_row  in T_Rt4x4:
            for T_col in T_row:
                T_4x4_list_i.append(T_col)
        T_4x4_list.append(T_4x4_list_i)
        #print(image_id,T_Rt4x4)
        
    with open(txt_name, 'w') as file:
        for row in T_4x4_list:
            
            line = ' '.join(map(str, row))
            file.write(f"{line}\n")
  
    #np.savetxt(out_txt, T_4x4_list)
    return T_4x4_list
        
    # Call the rendering script with the extrinsics file
    #subprocess.run(['python', render_script_path, '--extrinsics', extrinsics_file, '--output', f'{output_dir}/rendered_{image_id}.png'])

if __name__ == "__main__":

    path_txt="0测试数据1/d1_100mRTKColmap/sparse/0/"
    colmap_images_path = path_txt+"images.txt"

   
    camera_poses_T_4x4_output_dir = 'data/output/T_Rt4x4_.txt'
   
   
    camera_poses_q_t = read_colmap_images_file(colmap_images_path)
    T_4x4_list=images_pose_T4x4(camera_poses_q_t,camera_poses_T_4x4_output_dir)