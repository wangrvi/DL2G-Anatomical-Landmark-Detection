import glob
import os
import pickle
import numpy as np
import torch
import yaml
import SimpleITK as sitk
import torch.nn.functional as F


# path utils
def load_config_from_yaml(file, name=None):
    with open(file, 'r') as file:  
        data = yaml.safe_load(file)  
    if name is not None and name in data:
        return data[name]
    return data

def load_lists_from_pkls(pkl_paths):
    return [load_list_from_pkl(path) for path in pkl_paths]

def load_list_from_pkl(pklname):
    with open(pklname, 'rb') as file:
        data = pickle.load(file)
    return data

def get_pathlist_from_glob(repath):
    return sorted(glob.glob(repath))

def get_nameIn(path, position=-1):
    p, n = os.path.split(path)
    if position==0:
        return n.split('.')[0]   
    return p.split('/')[position]   

def checkPath_mkdirs(path):
    directory = os.path.dirname(path)  
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def save_list_to_pkl(list, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(list, f)

# tensor / numpy utils
def tensor2numpy_squeeze0(tensor):
    return tensor.squeeze(0).numpy()


# aug utils
def transMatrix(scale, tx, ty, tz, ax, ay, az):
    
    rx, ry, rz = np.deg2rad(ax), np.deg2rad(ay), np.deg2rad(az)  
      
    # 创建旋转矩阵  
    Rx = np.array([[1, 0, 0],  
                    [0, np.cos(rx), -np.sin(rx)],  
                    [0, np.sin(rx), np.cos(rx)]])  
      
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],  
                    [0, 1, 0],  
                    [-np.sin(ry), 0, np.cos(ry)]])  
      
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],  
                    [np.sin(rz), np.cos(rz), 0],  
                    [0, 0, 1]])  
      
    # 组合旋转矩阵（注意顺序）  
    R = np.dot(Rz, np.dot(Ry, Rx)).astype(np.float32)  
    
    S = np.array([[2-scale, 0, 0],  
                   [0, 2-scale, 0],  
                   [0, 0, 2-scale],  
                   ], dtype=np.float32)  
    R = np.dot(R, S)
    
    t = np.array([tx, ty, tz], dtype=np.float32)
    return R, t

def get_CenterOfMass(array):
    valid_indexs = np.where(array>0)
    center_of_mass = np.array(valid_indexs).mean(axis=1)
    return center_of_mass

def ImageArray_3DTransformeWithPoint(nparray, points, RotationScale=np.eye(3), scale=1, trans=np.zeros(3), center=None):
    '''
    对图像和图像的点进行对应的变换（旋转、缩放、平移）
    input：
        nparray： H, W, D
        points: n, 3
        RotationScale: 3, 3
        scale: 1
        trans: 1, 3


    1. 生成旋转缩放矩阵
    2. 基于torch.nn.functional.grid,使用旋转缩放矩阵对图像以及点进行变换
    3. 生成平移向量
    4. 基于SimpleITK生成平移的变换，并使用sitk.Resample对图像进行平移
    
    
    注释2： numpy中数组的存储是（z, y, x） 因此需要先将numpy的数组进行转置，然后再送入torch或者sitk中进行变换处理
    注释3： 在转换到numpy之后需要再转换成（x, y, z）
    注释4： sitk中的平移可能因为方向的问题，在Depth轴的方向平移分量点和图像是相反的
    注释5： sitk中的缩放需要尺度对于图像和点来说也是相反的        （有待测试）
    旋转：逆时针方向为旋转正方向
    注释1：旋转需要对点同时变换，但是需要注意点的变换矩阵和torch中的grid方向相反，因此两者的变换互为转置矩阵 见1
    '''
    # 图像的基本设置
    if center is None:
        center = get_CenterOfMass(nparray)
    ras_direction = [-1, 0, 0, 0, -1, 0, 0, 0, 1]  # 默认数组代表的图像方向，在SITK中默认LPS，因此需要设定RAS的方向
    
    # 对图像做旋转变换
    nparray = nparray.transpose(2,1,0).astype(np.float32)  # 注释 2
    tensor_array = torch.tensor(nparray).unsqueeze(0).unsqueeze(0) 
    size =[1, 1, *nparray.shape]
    RotationScale_M = np.hstack([RotationScale, np.zeros((3,1))]).astype(np.float32)
    RotationScale_M = np.vstack([RotationScale_M, np.array([0, 0, 0, 1])]).astype(np.float32)
    RotationScale_M = RotationScale_M.T  # 注释 1
    assert RotationScale_M.shape == (4, 4)
    grid = F.affine_grid(torch.tensor(RotationScale_M[0:3]).unsqueeze(0), size, align_corners=True) # matrix.shape=(1,3,4), size=(N, C, W, H, D)
    rotationScale_array = F.grid_sample(tensor_array, grid, mode='bilinear', align_corners=True)
    rotationScale_array = rotationScale_array.numpy().squeeze()
    rotationScale_array = rotationScale_array.transpose(2,1,0)  # 注释 3
    
    # 对点进行旋转变换 以及缩放
    rot_offset = np.array(center) - RotationScale/(2- scale) * scale @ np.array(center).T 
    rotationScale_points = (RotationScale/(2- scale) * scale @ points.T).T + rot_offset

    # 对图像进行平移\缩放变换（旋转之后）
    sitk_image = sitk.GetImageFromArray(rotationScale_array.transpose(2,1,0))  # 注释 2
    sitk_image.SetDirection(ras_direction)

    # scale_matrix = np.eye(3) * (1 + 1 - scale)
    # print(trans)
    trans = trans * np.array([1,1,-1])
    translate_trans = sitk.AffineTransform(np.eye(3).flatten().astype(np.float32).tolist(), trans.tolist())
    RotScaleTranslate_image = sitk.Resample(
        sitk_image, 
        translate_trans,
    )
    sitkArray = sitk.GetArrayFromImage(RotScaleTranslate_image)
    sitkArray = sitkArray.transpose(2, 1, 0)
    # 对点进行平移
    trans_offset =  trans * np.array([1,1,-1]) # 注释 4，注释5
    RotScaleTranslate_points =  rotationScale_points + trans_offset                    
    # scale_offset =  trans*np.array([1,1,-1]) + np.array(center) - (scale_matrix / (1 + 1 - scale) * scale) @ np.array(center).T  # 注释 4，注释5
    # RotScaleTranslate_points = ( (scale_matrix / (1 + 1 - scale) * scale) @ rotationScale_points.T).T  + scale_offset                    
    # scale_offset =  trans*np.array([1,1,-1]) + np.array(center) - scale_matrix/(matrix_para['scale']**2) @ np.array(center).T  # 注释 4，注释5
    # RotScaleTranslate_points = (scale_matrix/(matrix_para['scale']**2) @ rotationScale_points.T).T  + scale_offset                    
    
    
    # plot1array(nparray.transpose(2,1,0), givenPoint=points[0])
    # plot1array(rotationScale_array, givenPoint=rotationScale_points[0])
    # plot1array(sitkArray, givenPoint=RotScaleTranslate_points[0])
    return sitkArray, RotScaleTranslate_points, center


# dataset utils
def validPoints_Select(templateafids, validpoints, threshold=30):
    '''
    local select valid points
    input: 
        templateafids: fidnum * 3
        validpoints:  ImgNum * n * 3
    output:
        select_validPoints: ImgNum * selectNum * 3
    '''
    templateafids = np.array(templateafids)
    select_validPoints = []
 
    for sub_validpoints in validpoints:
        sub_validpoints = np.array(sub_validpoints)
        dis = np.linalg.norm(templateafids[:, np.newaxis] - sub_validpoints, axis=2)

        indexs = dis <= threshold
 
        indexs = np.sum(indexs, axis=0) > 0

        sub_selectPoints = sub_validpoints[indexs, :]

        select_validPoints.append(sub_selectPoints.tolist())
    return select_validPoints