# -*- coding: utf-8 -*-
# conda activate py37gaosi  # 服务器
# activate py38  # 笔记本

import os
import numpy as np

'''
# 原始gnss输入 四个数据
名字 纬度 经度 高度
DJI_0002.JPG 34.032505638888885 108.76779925 514.638
DJI_0005.JPG 34.03267641666667 108.76781155555555 514.464
DJI_0011.JPG 34.03394725 108.76789833333333 514.635

转化为  三个数据
纬度 经度 高度
34.032505638888885 108.76779925 514.638
34.03267641666667 108.76781155555555 514.464
34.03394725 108.76789833333333 514.635

'''
def API_data0123_to_data123(data0123):

    data123=[]
    for data_i in data0123:

        data_0=float(data_i[1])
        data_1=float(data_i[2])
        data_2=float(data_i[3])
        data_ii=[data_0,data_1,data_2]
        data123.append(data_ii)
    return data123

# 遍历文件夹读取 文件名字
def API_read_file_list(img_path_dir):

    file_dir_name_list=[]
   
    for filename in os.listdir(img_path_dir):
        file_dir_name=img_path_dir+filename

        
        file_dir_name_list.append(file_dir_name)
    

      
    return file_dir_name_list


def API_Save2txt(txt_name,Gnss_list):

    with open(txt_name, 'w') as file:
        for row in Gnss_list:
            line = ' '.join(map(str, row))
            file.write(f"{line}\n")

    print(txt_name,"保存成功")


def API_read2txt(txt_name):
    
    print(txt_name,"读取txt数据成功")
    Gnss_list = []
    with open(txt_name, 'r') as file:
        for line in file:
            row = list(map(str, line.split()))
            Gnss_list.append(row)
            #print(row)
    return Gnss_list

# def API_txt_to_Draw3D(list_name_xyz):
   
    
#     x_list=[]
#     y_list=[]
#     z_list=[]
#     for data_i in list_name_xyz:
#         nam_i=data_i[0]
#         x_i=float(data_i[1])
#         y_i=float(data_i[2])
#         z_i=float(data_i[3])
#         x_list.append(x_i)
#         y_list.append(y_i)
#         z_list.append(z_i)
#     return x_list,y_list,z_list
   


#====================测试========================
'''
if __name__ == "__main__":
   

    # 参数
    # 0-1 gps照片路径
    img_path_dir="E:/v0_Project/V0_Mybao/v8_slam/python工具/0测试数据/d1_100mRTKColmap/images/gps_images/"
    # 0-2 txt保存的名字
    GPS_txt_name="GPS.txt"

    # 1读取数据
    Gnss_list=API_read_directory(img_path_dir)

    # 2保存txt
    API_Save2txt(GPS_txt_name,Gnss_list)

    # 3读取txt
    Gnss_list_Read = API_read2txt(GPS_txt_name)

'''