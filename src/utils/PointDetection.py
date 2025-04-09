import numpy as np
import math
import random
import torchio as tio
import time
import torch
import sys
sys.path.append('/home/wangrui/code/autoFidDetection/DL2G')
from src.utils.util import get_nameIn


# neet to plot the feature points detection result to check the quality of feature points detection

def Get_FeaturePoints(args, datapaths):
    '''
    feature points detection for datapaths

    input:
        datapaths: list of str, the path of the data array(.npy)

    args params：
        datapaths: list of str, the path of the data
        resample_scale  (default: 4): scaled img to detection feature points
        mintranslate (default: 2): feature points augmentation min distance
        maxtranslate (default: 10): feature points augmentation max distance
        augnum (default: 6): feature points augmentation num
        preScreenSize (default: 1): preScreenSize
        screenAug (default: True): whether to use feature points augmentation
    
        note: default params are setting for OASIS and HCP dataset
    
    output:
        Data_PointLists: list of list, the feature points of the data
        screen_times: float, the mean time of feature points detection
    '''
    resample_scale = args.resampleScale
    augnum = args.augnum
    mintranslate = args.mintranslate
    maxtranslate = args.maxtranslate
    preScreenSize = args.preScreenSize
    Aug = args.screenAug

    # Get the pre-screening and initialization functions
    PreScreen = init_PreScreen(args)
    resample = tio.Resample(resample_scale)
    augpoint = AugPoints(augnum, maxtranslate, mintranslate)
    
    # Process each data item in a loop
    feature_points_num = 0
    screen_times = 0
    Data_PointLists = []
    for datapath in datapaths:
        # Perform initial screening of feature points on the data
            # First, downsample the data
        array = np.load(datapath)
        image = tio.ScalarImage(tensor=torch.tensor(array).unsqueeze(0))
        resample_image = resample(image)
        screen_array = tensor2numpy_squeeze0(resample_image.data)

        # Time recording
        end = time.time()
            ## Perform initial screening
        feature_points = PreScreen(screen_array, preScreenSize)
        feature_points = (np.array(feature_points) * resample_scale)

        if Aug:
        # Augment the initially screened feature points
            feature_points_aug = augpoint(feature_points)
                # Remove duplicates and perform range checking on the augmented feature points
            feature_points = constraint_Points_coordinate(feature_points_aug)
        # Stop time recording
        screen_times += time.time() - end
        print(f"{get_nameIn(datapath)}get feature point: {len(feature_points)}")

        feature_points_num += len(feature_points)
        # Add feature points
        Data_PointLists.append(feature_points)
    print("mean feature point nums:", feature_points_num/len(datapaths))
    print(f"{len(datapaths)} Image feature point detection mean time:", screen_times/len(datapaths))
    return Data_PointLists, screen_times/len(datapaths)


# Feature point initial screening code
def init_PreScreen(args):
    '''
    Initialize the pre-screening function according to the parameters.
    args formal parameters:
        Type of pre-screening.
        Parameters for each type of pre-screening.
    '''
    PreScreen = PreScreen_3d(args.constMean, SaddleFace=args.SaddleFace, SaddleAngle=args.SaddleAngle, SaddleArris=args.SaddleArris, CornerT=args.CornerT, CornerU=args.CornerU, CornerTMinus=args.CornerTMinus, CornerUMinus=args.CornerUMinus, Cross=args.Cross,\
                            thresh_FaceDVoxelNum1=args.thresh_FaceDVoxelNum1, thresh_FaceDVoxelNum2=args.thresh_FaceDVoxelNum2, thresh_AngleDVoxelNum1=args.thresh_AngleDVoxelNum1, thresh_AngleDVoxelNum2=args.thresh_AngleDVoxelNum2, thresh_ArrisDVoxelNum1=args.thresh_ArrisDVoxelNum1, thresh_ArrisDVoxelNum2=args.thresh_ArrisDVoxelNum2, \
                            thresh_UtypeMoreDVoxelNum1=args.thresh_UtypeMoreDVoxelNum1, thresh_UtypeMoreDVoxelNum2=args.thresh_UtypeMoreDVoxelNum2, thresh_UtypeLessDVoxelNum1=args.thresh_UtypeLessDVoxelNum1, thresh_UtypeLessDVoxelNum2=args.thresh_UtypeLessDVoxelNum2, \
                            thresh_UtypeMoreDVoxelNum1_=args.thresh_UtypeMoreDVoxelNum1_, thresh_UtypeMoreDVoxelNum2_=args.thresh_UtypeMoreDVoxelNum2_, thresh_UtypeLessDVoxelNum1_=args.thresh_UtypeLessDVoxelNum1_, thresh_UtypeLessDVoxelNum2_=args.thresh_UtypeLessDVoxelNum2_,\
                            thresh_TtypeMoreDVoxelNum1=args.thresh_TtypeMoreDVoxelNum1, thresh_TtypeMoreDVoxelNum2=args.thresh_TtypeMoreDVoxelNum2, thresh_TtypeLessDVoxelNum1=args.thresh_TtypeLessDVoxelNum1, thresh_TtypeLessDVoxelNum2=args.thresh_TtypeLessDVoxelNum2, \
                            thresh_TtypeMoreDVoxelNum1_=args.thresh_TtypeMoreDVoxelNum1_, thresh_TtypeMoreDVoxelNum2_=args.thresh_TtypeMoreDVoxelNum2_, thresh_TtypeLessDVoxelNum1_=args.thresh_TtypeLessDVoxelNum1_, thresh_TtypeLessDVoxelNum2_=args.thresh_TtypeLessDVoxelNum2_,\
                            thresh_CrossUpNum=args.thresh_CrossUpNum, thresh_CrossLeftNum=args.thresh_CrossLeftNum, thresh_CrossRightNum=args.thresh_CrossRightNum, thresh_CrossMiddleNum=args.thresh_CrossMiddleNum)
    return PreScreen
class PreScreen_3d():
    def __init__(self, threshold, SaddleFace=False, SaddleAngle=False, SaddleArris=False, CornerT=False, CornerU=False, CornerTMinus=False, CornerUMinus=False, Cross=False,\
                 thresh_FaceDVoxelNum1=0, thresh_FaceDVoxelNum2=3, thresh_AngleDVoxelNum1=0, thresh_AngleDVoxelNum2=3, thresh_ArrisDVoxelNum1=0, thresh_ArrisDVoxelNum2=2, \
                 thresh_UtypeMoreDVoxelNum1=0, thresh_UtypeMoreDVoxelNum2=17, thresh_UtypeLessDVoxelNum1=17, thresh_UtypeLessDVoxelNum2=0, \
                 thresh_UtypeMoreDVoxelNum1_=0, thresh_UtypeMoreDVoxelNum2_=19, thresh_UtypeLessDVoxelNum1_=19, thresh_UtypeLessDVoxelNum2_=0,\
                 thresh_TtypeMoreDVoxelNum1=0, thresh_TtypeMoreDVoxelNum2=14, thresh_TtypeLessDVoxelNum1=0, thresh_TtypeLessDVoxelNum2=12, \
                 thresh_TtypeMoreDVoxelNum1_=2, thresh_TtypeMoreDVoxelNum2_=14, thresh_TtypeLessDVoxelNum1_=2, thresh_TtypeLessDVoxelNum2_=12,\
                 thresh_CrossUpNum=0, thresh_CrossLeftNum=0, thresh_CrossRightNum=1, thresh_CrossMiddleNum=1):
        # This parameter is the mean of the dataset, or a value set for each individual data.
        self.threshold = threshold

        self.SaddleFace = SaddleFace
        self.SaddleAngle = SaddleAngle
        self.SaddleArris = SaddleArris
        self.CornerT = CornerT
        self.CornerU = CornerU
        self.CornerTMinus = CornerTMinus
        self.CornerUMinus = CornerUMinus
        self.Cross = Cross
        
        if self.SaddleFace:
            self.thresh_FaceDVoxelNum1 = thresh_FaceDVoxelNum1
            self.thresh_FaceDVoxelNum2 = thresh_FaceDVoxelNum2

        if self.SaddleAngle:
           
            self.thresh_AngleDVoxelNum1 = thresh_AngleDVoxelNum1

            self.thresh_AngleDVoxelNum2 = thresh_AngleDVoxelNum2
        
        if self.SaddleArris:
            self.thresh_ArrisDVoxelNum1 = thresh_ArrisDVoxelNum1

            self.thresh_ArrisDVoxelNum2 = thresh_ArrisDVoxelNum2

        if self.CornerU:
            self.thresh_UtypeMoreDVoxelNum1 = thresh_UtypeMoreDVoxelNum1

            self.thresh_UtypeMoreDVoxelNum2 = thresh_UtypeMoreDVoxelNum2  
            self.thresh_UtypeLessDVoxelNum1 = thresh_UtypeLessDVoxelNum1

            self.thresh_UtypeLessDVoxelNum2 = thresh_UtypeLessDVoxelNum2  
        if self.CornerUMinus:
            self.thresh_UtypeMoreDVoxelNum1 = thresh_UtypeMoreDVoxelNum1

            self.thresh_UtypeMoreDVoxelNum2 = thresh_UtypeMoreDVoxelNum2  
            self.thresh_UtypeLessDVoxelNum1 = thresh_UtypeLessDVoxelNum1

            self.thresh_UtypeLessDVoxelNum2 = thresh_UtypeLessDVoxelNum2  
            self.thresh_UtypeMoreDVoxelNum1_ = thresh_UtypeMoreDVoxelNum1_

            self.thresh_UtypeMoreDVoxelNum2_ = thresh_UtypeMoreDVoxelNum2_  
            self.thresh_UtypeLessDVoxelNum1_ = thresh_UtypeLessDVoxelNum1_

            self.thresh_UtypeLessDVoxelNum2_ = thresh_UtypeLessDVoxelNum2_  
        if self.CornerT:
            self.thresh_TtypeMoreDVoxelNum1 = thresh_TtypeMoreDVoxelNum1

            self.thresh_TtypeMoreDVoxelNum2 = thresh_TtypeMoreDVoxelNum2  
            self.thresh_TtypeLessDVoxelNum1 = thresh_TtypeLessDVoxelNum1

            self.thresh_TtypeLessDVoxelNum2 = thresh_TtypeLessDVoxelNum2  
        if self.CornerTMinus:
            self.thresh_TtypeMoreDVoxelNum1 = thresh_TtypeMoreDVoxelNum1

            self.thresh_TtypeMoreDVoxelNum2 = thresh_TtypeMoreDVoxelNum2  
            self.thresh_TtypeLessDVoxelNum1 = thresh_TtypeLessDVoxelNum1

            self.thresh_TtypeLessDVoxelNum2 = thresh_TtypeLessDVoxelNum2  
            self.thresh_TtypeMoreDVoxelNum1_ = thresh_TtypeMoreDVoxelNum1_

            self.thresh_TtypeMoreDVoxelNum2_ = thresh_TtypeMoreDVoxelNum2_  
            self.thresh_TtypeLessDVoxelNum1_ = thresh_TtypeLessDVoxelNum1_

            self.thresh_TtypeLessDVoxelNum2_ = thresh_TtypeLessDVoxelNum2_  
                     
        if self.Cross:
            self.thresh_CrossUpNum = thresh_CrossUpNum  
            self.thresh_CrossLeftNum = thresh_CrossLeftNum  
            self.thresh_CrossRightNum = thresh_CrossRightNum  
            self.thresh_CrossMiddleNum = thresh_CrossMiddleNum  
        
    def __call__(self, data, size):
    
        dim = data.shape
        
        start_i = size
        end_i = dim[0] - size 
        start_j = size
        end_j = dim[1] - size 
        start_k = size
        end_k = dim[2] - size 
        
    
        point_1 = data[start_i-size:end_i-size, start_j-size:end_j-size, start_k+size:end_k+size] 
        point_2 = data[start_i:end_i, start_j-size:end_j-size, start_k+size:end_k+size]
        point_3 = data[start_i+size:end_i+size, start_j-size:end_j-size, start_k+size:end_k+size] 
        point_4 = data[start_i-size:end_i-size, start_j:end_j, start_k+size:end_k+size]
        point_5 = data[start_i:end_i, start_j:end_j, start_k+size:end_k+size]
        point_6 = data[start_i+size:end_i+size, start_j:end_j, start_k+size:end_k+size]
        point_7 = data[start_i-size:end_i-size, start_j+size:end_j+size, start_k+size:end_k+size] 
        point_8 = data[start_i:end_i, start_j+size:end_j+size, start_k+size:end_k+size]
        point_9 = data[start_i+size:end_i+size, start_j+size:end_j+size, start_k+size:end_k+size] 
        point_10 = data[start_i-size:end_i-size, start_j-size:end_j-size, start_k:end_k]
        point_11 = data[start_i:end_i, start_j-size:end_j-size, start_k:end_k]
        point_12 = data[start_i+size:end_i+size, start_j-size:end_j-size, start_k:end_k]
        point_13 = data[start_i-size:end_i-size, start_j:end_j, start_k:end_k]
        point_14 = data[start_i:end_i, start_j:end_j, start_k:end_k]
        point_15 = data[start_i+size:end_i+size, start_j:end_j, start_k:end_k]
        point_16 = data[start_i-size:end_i-size, start_j+size:end_j+size, start_k:end_k]
        point_17 = data[start_i:end_i, start_j+size:end_j+size, start_k:end_k]
        point_18 = data[start_i+size:end_i+size, start_j+size:end_j+size, start_k:end_k]
        point_19 = data[start_i-size:end_i-size, start_j-size:end_j-size, start_k-size:end_k-size] 
        point_20 = data[start_i:end_i, start_j-size:end_j-size, start_k-size:end_k-size]
        point_21 = data[start_i+size:end_i+size, start_j-size:end_j-size, start_k-size:end_k-size] 
        point_22 = data[start_i-size:end_i-size, start_j:end_j, start_k-size:end_k-size]
        point_23 = data[start_i:end_i, start_j:end_j, start_k-size:end_k-size]
        point_24 = data[start_i+size:end_i+size, start_j:end_j, start_k-size:end_k-size]
        point_25 = data[start_i-size:end_i-size, start_j+size:end_j+size, start_k-size:end_k-size] 
        point_26 = data[start_i:end_i, start_j+size:end_j+size, start_k-size:end_k-size]
        point_27 = data[start_i+size:end_i+size, start_j+size:end_j+size, start_k-size:end_k-size] 
        
        if self.SaddleFace:
 
            # six face area (front, back, left, right, up, down)
            faceArea_front = self.build_blocks(point_8, point_16, point_17, point_18, point_26)
            faceArea_back = self.build_blocks(point_2, point_10, point_11, point_12, point_20)
            faceArea_left = self.build_blocks(point_4, point_10, point_13, point_16, point_22)
            faceArea_right = self.build_blocks(point_6, point_12, point_15, point_18, point_24)
            faceArea_up = self.build_blocks(point_8, point_4, point_5, point_6, point_2)
            faceArea_down = self.build_blocks(point_26, point_22, point_23, point_24, point_20)
        if self.SaddleAngle:

            # eight angle area (front left up, front right up, front right down, front left down, back left up, back right up, back right down, back left down)
            angleArea_FrontLeftUp = self.build_blocks(point_7, point_4, point_8, point_16)
            angleArea_FrontRightUp = self.build_blocks(point_9, point_6, point_8, point_18)
            angleArea_FrontRightDown = self.build_blocks(point_27, point_24, point_26, point_18)
            angleArea_FrontLeftDown = self.build_blocks(point_25, point_22, point_26, point_16)
            angleArea_BackLeftUp = self.build_blocks(point_1, point_4, point_2, point_10)
            angleArea_BackRightUp = self.build_blocks(point_3, point_2, point_6, point_12)
            angleArea_BackRightDown = self.build_blocks(point_21, point_20, point_12, point_24)
            angleArea_BackLeftDown = self.build_blocks(point_19, point_20, point_22, point_10)
        if self.SaddleArris:

            # eight diagonal edge area (front up, front down, back up, back down, left up, left down, right up, right down)
            arrisArea_FontUp = self.build_blocks(point_7, point_8, point_9)
            arrisArea_FontDown = self.build_blocks(point_25, point_26, point_27)
            arrisArea_BackUp = self.build_blocks(point_1, point_2, point_3)
            arrisArea_BackDown = self.build_blocks(point_19, point_20, point_21)
            arrisArea_LeftUp = self.build_blocks(point_1, point_4, point_7)
            arrisArea_LeftDown = self.build_blocks(point_19, point_22, point_25)
            arrisArea_RightUp = self.build_blocks(point_3, point_6, point_9)
            arrisArea_RightDown = self.build_blocks(point_21, point_24, point_27)
            

        # three center slices (coronal face, sagittal face, transverse face)
        centerface_coronal = self.build_blocks(point_4, point_5, point_6, point_13, point_15, point_22, point_23, point_24)
        centerface_sagittal = self.build_blocks(point_8, point_5, point_2, point_17, point_11, point_26, point_23, point_20)
        centerface_transection = self.build_blocks(point_16, point_17, point_18, point_13, point_15, point_10, point_11, point_12)

        # before and after long bars (up 123, middle 123, back 123)
        FBstripArea_Up1 = self.build_blocks(point_1, point_4, point_7)
        FBstripArea_Up2 = self.build_blocks(point_2, point_5, point_8)
        FBstripArea_Up3 = self.build_blocks(point_3, point_6, point_9)
        FBstripArea_Middle1 = self.build_blocks(point_10, point_13, point_16)
        FBstripArea_Middle2 = self.build_blocks(point_11, point_14, point_17)
        FBstripArea_Middle3 = self.build_blocks(point_12, point_15, point_18)
        FBstripArea_Down1 = self.build_blocks(point_19, point_22, point_25)
        FBstripArea_Down2 = self.build_blocks(point_20, point_23, point_26)
        FBstripArea_Down3 = self.build_blocks(point_21, point_24, point_27)

        # left and right long bars (up 123, middle 123, down 123)
        LRstripArea_Up1 = self.build_blocks(point_1, point_2, point_3)
        LRstripArea_Up2 = self.build_blocks(point_4, point_5, point_6)
        LRstripArea_Up3 = self.build_blocks(point_7, point_8, point_9)
        LRstripArea_Middle1 = self.build_blocks(point_10, point_11, point_12)
        LRstripArea_Middle2 = self.build_blocks(point_13, point_14, point_15)
        LRstripArea_Middle3 = self.build_blocks(point_16, point_17, point_18)
        LRstripArea_Down1 = self.build_blocks(point_19, point_20, point_21)
        LRstripArea_Down2 = self.build_blocks(point_22, point_23, point_24)
        LRstripArea_Down3 = self.build_blocks(point_25, point_26, point_27)
        

        result_list = []
        if self.SaddleFace:
            result_list.append(self.saddle_FaceDecision(faceArea_front, faceArea_back, faceArea_left, faceArea_right, faceArea_up, faceArea_down, centerface_coronal, centerface_sagittal, centerface_transection))
        if self.SaddleAngle:
            result_list.append(self.saddle_AngleDecision(angleArea_FrontLeftUp, angleArea_FrontRightUp, angleArea_FrontRightDown, angleArea_FrontLeftDown, angleArea_BackLeftUp, angleArea_BackRightUp, angleArea_BackRightDown, angleArea_BackLeftDown))
        if self.SaddleArris:
            result_list.append(self.saddle_ArrisDecision(arrisArea_FontUp, arrisArea_FontDown, arrisArea_BackUp, arrisArea_BackDown, arrisArea_LeftUp, arrisArea_LeftDown, arrisArea_RightUp, arrisArea_RightDown))
        if self.CornerT:
            result_list.append(self.corner_TtypeDecision(FBstripArea_Up1, FBstripArea_Up2, FBstripArea_Up3, FBstripArea_Middle1, FBstripArea_Middle2, FBstripArea_Middle3, FBstripArea_Down1, FBstripArea_Down2, FBstripArea_Down3, \
                                                                    LRstripArea_Up1, LRstripArea_Up2, LRstripArea_Up3, LRstripArea_Middle1, LRstripArea_Middle2, LRstripArea_Middle3, LRstripArea_Down1, LRstripArea_Down2, LRstripArea_Down3))
        if self.CornerU:
            result_list.append(self.corner_UtypeDecision(FBstripArea_Up1, FBstripArea_Up2, FBstripArea_Up3, FBstripArea_Middle1, FBstripArea_Middle2, FBstripArea_Middle3, FBstripArea_Down1, FBstripArea_Down2, FBstripArea_Down3, LRstripArea_Middle1, LRstripArea_Middle2, LRstripArea_Middle3))
        if self.Cross:
            result_list.append(self.cross_Decision(point_5, FBstripArea_Middle2, FBstripArea_Down2, point_13, point_15, point_22, point_24, FBstripArea_Up1, FBstripArea_Up3, point_2, point_8, point_16, point_25, point_18, point_27, point_12, point_21, point_10, point_19))
        if self.CornerUMinus:
            corner_UtypeDecision_result = self.corner_UtypeDecision(FBstripArea_Up1, FBstripArea_Up2, FBstripArea_Up3, FBstripArea_Middle1, FBstripArea_Middle2, FBstripArea_Middle3, FBstripArea_Down1, FBstripArea_Down2, FBstripArea_Down3, LRstripArea_Middle1, LRstripArea_Middle2, LRstripArea_Middle3)
            corner_UtypeDecision_result_ = self.corner_UtypeDecision(FBstripArea_Up1, FBstripArea_Up2, FBstripArea_Up3, FBstripArea_Middle1, FBstripArea_Middle2, FBstripArea_Middle3, FBstripArea_Down1, FBstripArea_Down2, FBstripArea_Down3, LRstripArea_Middle1, LRstripArea_Middle2, LRstripArea_Middle3, minus_param=True)
            result_list.append(np.all(np.stack((corner_UtypeDecision_result, ~corner_UtypeDecision_result_), axis=3), axis=3))
        if self.CornerTMinus:
            corner_TtypeDecision_result = self.corner_TtypeDecision(FBstripArea_Up1, FBstripArea_Up2, FBstripArea_Up3, FBstripArea_Middle1, FBstripArea_Middle2, FBstripArea_Middle3, FBstripArea_Down1, FBstripArea_Down2, FBstripArea_Down3, \
                                                                    LRstripArea_Up1, LRstripArea_Up2, LRstripArea_Up3, LRstripArea_Middle1, LRstripArea_Middle2, LRstripArea_Middle3, LRstripArea_Down1, LRstripArea_Down2, LRstripArea_Down3)
            corner_TtypeDecision_result_ = self.corner_TtypeDecision(FBstripArea_Up1, FBstripArea_Up2, FBstripArea_Up3, FBstripArea_Middle1, FBstripArea_Middle2, FBstripArea_Middle3, FBstripArea_Down1, FBstripArea_Down2, FBstripArea_Down3, \
                                                                    LRstripArea_Up1, LRstripArea_Up2, LRstripArea_Up3, LRstripArea_Middle1, LRstripArea_Middle2, LRstripArea_Middle3, LRstripArea_Down1, LRstripArea_Down2, LRstripArea_Down3, minus_param=True)
            result_list.append(np.all(np.stack((corner_TtypeDecision_result, ~corner_TtypeDecision_result_), axis=3), axis=3))
        result = np.any(np.stack(result_list, axis=3), axis=3)
        point_indexes = np.argwhere(result==True)
        return point_indexes.tolist()
    def cross_Decision(self, point_5, FBstripArea_Middle2, FBstripArea_Down2, point_13, point_15, point_22, point_24, FBstripArea_Up1, FBstripArea_Up3, point_2, point_8, point_16, point_25, point_18, point_27, point_12, point_21, point_10, point_19):
        part_outer = np.stack((point_2, point_8, point_16, point_25, point_18, point_27, point_12, point_21, point_10, point_19))
        cross_outer = np.concatenate((FBstripArea_Up1, FBstripArea_Up3, part_outer))
        cross_UpSubOuter = self.Screen_twoAreaCompare(point_5, cross_outer, CrossUp=True)

        cross_left = np.stack((point_13, point_22))
        cross_LeftSubOuter = self.Screen_twoAreaCompare(cross_left, cross_outer, CrossLeft=True)
        cross_Right = np.stack((point_15, point_24))
        cross_RightSubOuter = self.Screen_twoAreaCompare(cross_Right, cross_outer, CrossRight=True)
        cross_Middle = np.concatenate((FBstripArea_Middle2, FBstripArea_Down2))
        cross_MiddleSubOuter = self.Screen_twoAreaCompare(cross_Middle, cross_outer, CrossMiddle=True)

        result = np.all(np.stack((cross_UpSubOuter, cross_LeftSubOuter, cross_RightSubOuter, cross_MiddleSubOuter), axis=3), axis=3)
        return result
        
        
    def saddle_FaceDecision(self, faceArea_front, faceArea_back, faceArea_left, faceArea_right, faceArea_up, faceArea_down, centerface_coronal, centerface_sagittal, centerface_transection):
        fontBackResult = self.saddle_FaceCompare(faceArea_front, faceArea_back, centerface_coronal)
        leftRightResult = self.saddle_FaceCompare(faceArea_left, faceArea_right, centerface_sagittal)
        upDownResult = self.saddle_FaceCompare(faceArea_up, faceArea_down, centerface_transection)
        result = np.any(np.stack((fontBackResult, leftRightResult, upDownResult), axis=3), axis=3)
        return result
    def corner_UtypeDecision(self, FBstripArea_Up1, FBstripArea_Up2, FBstripArea_Up3, FBstripArea_Middle1, FBstripArea_Middle2, FBstripArea_Middle3, FBstripArea_Down1, FBstripArea_Down2, FBstripArea_Down3, LRstripArea_Middle1, LRstripArea_Middle2, LRstripArea_Middle3, minus_param=None):
        downOuter = np.concatenate((FBstripArea_Up1, FBstripArea_Up2, FBstripArea_Up3, FBstripArea_Middle1, FBstripArea_Middle3, FBstripArea_Down1, FBstripArea_Down3))
        downInner = np.concatenate((FBstripArea_Middle2, FBstripArea_Down2))
        Utype_Down = self.corner_UtypeCompare(downOuter, downInner, minus_param)
        
        upOuter = np.concatenate((FBstripArea_Up1, FBstripArea_Down2, FBstripArea_Up3, FBstripArea_Middle1, FBstripArea_Middle3, FBstripArea_Down1, FBstripArea_Down3))
        upInner = np.concatenate((FBstripArea_Middle2, FBstripArea_Up2))
        Utype_Up = self.corner_UtypeCompare(upOuter, upInner, minus_param)
        
        leftOuter = np.concatenate((FBstripArea_Up1, FBstripArea_Up2, FBstripArea_Up3, FBstripArea_Down1, FBstripArea_Down2, FBstripArea_Down3, FBstripArea_Middle3))
        leftInner = np.concatenate((FBstripArea_Middle2, FBstripArea_Middle1))
        Utype_Left = self.corner_UtypeCompare(leftOuter, leftInner, minus_param)
        
        rightOuter = np.concatenate((FBstripArea_Up1, FBstripArea_Up2, FBstripArea_Up3, FBstripArea_Down1, FBstripArea_Down2, FBstripArea_Down3, FBstripArea_Middle1))
        rightInner = np.concatenate((FBstripArea_Middle2, FBstripArea_Middle3))
        Utype_Right = self.corner_UtypeCompare(rightOuter, rightInner, minus_param)
        
        FontOuter = np.concatenate((FBstripArea_Up1, FBstripArea_Up2, FBstripArea_Up3, FBstripArea_Down1, FBstripArea_Down2, FBstripArea_Down3, LRstripArea_Middle1))
        FontInner = np.concatenate((LRstripArea_Middle2, LRstripArea_Middle3))
        Utype_Font = self.corner_UtypeCompare(FontOuter, FontInner, minus_param)
        
        BackOuter = np.concatenate((FBstripArea_Up1, FBstripArea_Up2, FBstripArea_Up3, FBstripArea_Down1, FBstripArea_Down2, FBstripArea_Down3, LRstripArea_Middle3))
        BackInner = np.concatenate((LRstripArea_Middle2, LRstripArea_Middle1))
        Utype_Back = self.corner_UtypeCompare(BackOuter, BackInner, minus_param)

        result = np.any(np.stack((Utype_Down, Utype_Up, Utype_Left, Utype_Right, Utype_Font, Utype_Back), axis=3), axis=3)
        return result
    def corner_TtypeDecision(self, FBstripArea_Up1, FBstripArea_Up2, FBstripArea_Up3, FBstripArea_Middle1, FBstripArea_Middle2, FBstripArea_Middle3, FBstripArea_Down1, FBstripArea_Down2, FBstripArea_Down3, \
                             LRstripArea_Up1, LRstripArea_Up2, LRstripArea_Up3, LRstripArea_Middle1, LRstripArea_Middle2, LRstripArea_Middle3, LRstripArea_Down1, LRstripArea_Down2, LRstripArea_Down3, minus_param=None):
        FBRightDownOuter = np.concatenate((FBstripArea_Up1, FBstripArea_Up2, FBstripArea_Up3, FBstripArea_Middle1, FBstripArea_Down1))
        FBRightDownInner = np.concatenate((FBstripArea_Middle2, FBstripArea_Middle3, FBstripArea_Down2, FBstripArea_Down3))
        Ttype_FBRightDown = self.corner_TtypeCompare(FBRightDownOuter, FBRightDownInner, minus_param)
                                 
        FBRightUpOuter = np.concatenate((FBstripArea_Up1, FBstripArea_Middle1, FBstripArea_Down1, FBstripArea_Down2, FBstripArea_Down3))
        FBRightUpInner = np.concatenate((FBstripArea_Up2, FBstripArea_Up3, FBstripArea_Middle2, FBstripArea_Middle3))
        Ttype_FBRightUp = self.corner_TtypeCompare(FBRightUpOuter, FBRightUpInner, minus_param)                               
    
        FBLeftUpOuter = np.concatenate((FBstripArea_Up3, FBstripArea_Middle3, FBstripArea_Down1, FBstripArea_Down2, FBstripArea_Down3))
        FBLeftUpInner = np.concatenate((FBstripArea_Up1, FBstripArea_Up2, FBstripArea_Middle1, FBstripArea_Middle2))
        Ttype_FBLeftUp = self.corner_TtypeCompare(FBLeftUpOuter, FBLeftUpInner, minus_param)      
                                 
        FBLeftDownOuter = np.concatenate((FBstripArea_Up1, FBstripArea_Up2, FBstripArea_Up3, FBstripArea_Middle3, FBstripArea_Down3))
        FBLeftDownInner = np.concatenate((FBstripArea_Middle1, FBstripArea_Middle2, FBstripArea_Down1, FBstripArea_Down2))
        Ttype_FBLeftDown = self.corner_TtypeCompare(FBLeftDownOuter, FBLeftDownInner, minus_param)             
                                 
        LRFontDownOuter = np.concatenate((LRstripArea_Up1, LRstripArea_Up2, LRstripArea_Up3, LRstripArea_Middle1, LRstripArea_Down1))
        LRFontDownInner = np.concatenate((LRstripArea_Middle2, LRstripArea_Middle3, LRstripArea_Down2, LRstripArea_Down3))
        Ttype_LRFontDown = self.corner_TtypeCompare(LRFontDownOuter, LRFontDownInner, minus_param)
                                 
        LRFontUpOuter = np.concatenate((LRstripArea_Up1, LRstripArea_Middle1, LRstripArea_Down1, LRstripArea_Down2, LRstripArea_Down3))
        LRFontUpInner = np.concatenate((LRstripArea_Up2, LRstripArea_Up3, LRstripArea_Middle2, LRstripArea_Middle3))
        Ttype_LRFontUp = self.corner_TtypeCompare(LRFontUpOuter, LRFontUpInner, minus_param)                               
    
        LRBackUpOuter = np.concatenate((LRstripArea_Up3, LRstripArea_Middle3, LRstripArea_Down1, LRstripArea_Down2, LRstripArea_Down3))
        LRBackUpInner = np.concatenate((LRstripArea_Up1, LRstripArea_Up2, LRstripArea_Middle1, LRstripArea_Middle2))
        Ttype_LRBackUp = self.corner_TtypeCompare(LRBackUpOuter, LRBackUpInner, minus_param)      
                                 
        LRBackDownOuter = np.concatenate((LRstripArea_Up1, LRstripArea_Up2, LRstripArea_Up3, LRstripArea_Middle3, LRstripArea_Down3))
        LRBackDownInner = np.concatenate((LRstripArea_Middle1, LRstripArea_Middle2, LRstripArea_Down1, LRstripArea_Down2))
        Ttype_LRBackDown = self.corner_TtypeCompare(LRBackDownOuter, LRBackDownInner, minus_param)     

        result = np.any(np.stack((Ttype_FBRightDown, Ttype_FBRightUp, Ttype_FBLeftUp, Ttype_FBLeftDown, Ttype_LRFontDown, Ttype_LRFontUp, Ttype_LRBackUp, Ttype_LRBackDown), axis=3), axis=3)
        return result
    def saddle_ArrisDecision(self, arrisArea_FontUp, arrisArea_FontDown, arrisArea_BackUp, arrisArea_BackDown, arrisArea_LeftUp, arrisArea_LeftDown, arrisArea_RightUp, arrisArea_RightDown):
        FrontUp_BackDown_Result = self.saddle_ArrisCompare(arrisArea_FontUp, arrisArea_BackDown, arrisArea_FontDown, arrisArea_BackUp)
        FrontLeftDown_BackRightUp_Result = self.saddle_ArrisCompare(arrisArea_FontDown, arrisArea_BackUp, arrisArea_FontUp, arrisArea_BackDown)
        BackLeftUp_FrontRightDown_Result = self.saddle_ArrisCompare(arrisArea_LeftUp, arrisArea_RightDown, arrisArea_LeftDown, arrisArea_RightUp)
        BackLeftDown_FrontRightUp_Result = self.saddle_ArrisCompare(arrisArea_LeftDown, arrisArea_RightUp, arrisArea_LeftUp, arrisArea_RightDown)
        result = np.any(np.stack((FrontUp_BackDown_Result, FrontLeftDown_BackRightUp_Result, BackLeftUp_FrontRightDown_Result, BackLeftDown_FrontRightUp_Result), axis=3), axis=3)
        return result
        
    def saddle_AngleDecision(self, angleArea_FrontLeftUp, angleArea_FrontRightUp, angleArea_FrontRightDown, angleArea_FrontLeftDown, angleArea_BackLeftUp, angleArea_BackRightUp, angleArea_BackRightDown, angleArea_BackLeftDown):   
        FrontLeftUp_BackRightDown_Result = self.saddle_AngleCompare(angleArea_FrontLeftUp, angleArea_FrontLeftUp, angleArea_BackLeftUp, angleArea_FrontRightUp, angleArea_FrontRightDown, angleArea_BackLeftDown)
        FrontLeftDown_BackRightUp_Result = self.saddle_AngleCompare(angleArea_FrontLeftDown, angleArea_BackRightUp, angleArea_BackLeftUp, angleArea_FrontRightUp, angleArea_FrontRightDown, angleArea_BackLeftDown)
        BackLeftUp_FrontRightDown_Result = self.saddle_AngleCompare(angleArea_BackLeftUp, angleArea_FrontRightDown, angleArea_FrontLeftUp, angleArea_FrontLeftUp, angleArea_FrontLeftDown, angleArea_BackRightUp)
        BackLeftDown_FrontRightUp_Result = self.saddle_AngleCompare(angleArea_BackLeftDown, angleArea_FrontRightUp, angleArea_FrontLeftUp, angleArea_FrontLeftUp, angleArea_FrontLeftDown, angleArea_BackRightUp)

        result = np.any(np.stack((FrontLeftUp_BackRightDown_Result, FrontLeftDown_BackRightUp_Result, BackLeftUp_FrontRightDown_Result, BackLeftDown_FrontRightUp_Result), axis=3), axis=3)
        return result
    def corner_UtypeCompare(self, outer, inner, minus_param):
        if minus_param:
            OuterSubinner = self.Screen_twoAreaCompare(outer, inner, UtypeCompareOuter_=True)
            innerSubouter = self.Screen_twoAreaCompare(inner, outer, UtypeCompareInner_=True)
        else:
            OuterSubinner = self.Screen_twoAreaCompare(outer, inner, UtypeCompareOuter=True)
            innerSubouter = self.Screen_twoAreaCompare(inner, outer, UtypeCompareInner=True)
        jointResult = np.any(np.stack((OuterSubinner, innerSubouter), axis=3), axis=3)
        return jointResult
    def corner_TtypeCompare(self, outer, inner, minus_param):
        if minus_param:
            OuterSubinner = self.Screen_twoAreaCompare(outer, inner, TtypeCompareOuter_=True)
            innerSubouter = self.Screen_twoAreaCompare(inner, outer, TtypeCompareInner_=True)
        else:
            OuterSubinner = self.Screen_twoAreaCompare(outer, inner, TtypeCompareOuter=True)
            innerSubouter = self.Screen_twoAreaCompare(inner, outer, TtypeCompareInner=True)
        jointResult = np.any(np.stack((OuterSubinner, innerSubouter), axis=3), axis=3)
        return jointResult
    def saddle_ArrisCompare(self, mainArea1, mainArea2, middleArris1, middleArris2):
        middleArriss = np.concatenate((middleArris1, middleArris2), axis=0)
        main1SubMiddle = self.Screen_twoAreaCompare(mainArea1, middleArriss, ArrisCompare=True)
        main2SubMiddle = self.Screen_twoAreaCompare(mainArea2, middleArriss, ArrisCompare=True)
        jointResult = np.all(np.stack((main1SubMiddle, main2SubMiddle), axis=3), axis=3)
        return jointResult
        
    def saddle_AngleCompare(self, mainArea1, mainArea2, middleAngle1, middleAngle2, middleAngle3, middleAngle4):
        middleAngles = np.stack((middleAngle1, middleAngle2, middleAngle3, middleAngle4))
        main1SubMiddle = self.Screen_twoAreaCompare(mainArea1, middleAngles, AngleCompare=True)
        main2SubMiddle = self.Screen_twoAreaCompare(mainArea2, middleAngles, AngleCompare=True)
        jointResult = np.all(np.stack((main1SubMiddle, main2SubMiddle), axis=3), axis=3)
        return jointResult
        
    def saddle_FaceCompare(self, mainArea1, mainArea2, middleArea):
        main1SubMiddle = self.Screen_twoAreaCompare(mainArea1, middleArea, FaceCompare=True)
        main2SubMiddle = self.Screen_twoAreaCompare(mainArea2, middleArea, FaceCompare=True)
        jointResult = np.all(np.stack((main1SubMiddle, main2SubMiddle), axis=3), axis=3)
        return jointResult
    def Screen_twoAreaCompare(self, mainArea, secondArea, FaceCompare=None, AngleCompare=None, ArrisCompare=None, UtypeCompareOuter=None, UtypeCompareInner=None, UtypeCompareOuter_=None, UtypeCompareInner_=None,\
                              TtypeCompareOuter=None, TtypeCompareInner=None, TtypeCompareOuter_=None, TtypeCompareInner_=None, CrossUp=None, CrossLeft=None, CrossRight=None, CrossMiddle=None):
        if FaceCompare:
            twoAreaCompare = np.sum(np.sum((mainArea[:, np.newaxis] -  secondArea) > self.threshold, axis=1) > self.thresh_FaceDVoxelNum1, axis=0) > self.thresh_FaceDVoxelNum2
        if AngleCompare:
            twoAreaCompare = np.sum(np.sum((mainArea[:, np.newaxis] -  secondArea) > self.threshold, axis=1) > self.thresh_AngleDVoxelNum1, axis=0) > self.thresh_AngleDVoxelNum2
        if ArrisCompare:
            twoAreaCompare = np.sum(np.sum((mainArea[:, np.newaxis] -  secondArea) > self.threshold, axis=1) > self.thresh_ArrisDVoxelNum1, axis=0) > self.thresh_ArrisDVoxelNum2
        if UtypeCompareOuter:
            twoAreaCompare = np.sum(np.sum((mainArea[:, np.newaxis] -  secondArea) > self.threshold, axis=1) > self.thresh_UtypeMoreDVoxelNum1, axis=0) > self.thresh_UtypeMoreDVoxelNum2
        if UtypeCompareInner:
            twoAreaCompare = np.sum(np.sum((mainArea[:, np.newaxis] -  secondArea) > self.threshold, axis=1) > self.thresh_UtypeLessDVoxelNum1, axis=0) > self.thresh_UtypeLessDVoxelNum2
        if UtypeCompareOuter_:
            twoAreaCompare = np.sum(np.sum((mainArea[:, np.newaxis] -  secondArea) > self.threshold, axis=1) > self.thresh_UtypeMoreDVoxelNum1_, axis=0) > self.thresh_UtypeMoreDVoxelNum2_
        if UtypeCompareInner_:
            twoAreaCompare = np.sum(np.sum((mainArea[:, np.newaxis] -  secondArea) > self.threshold, axis=1) > self.thresh_UtypeLessDVoxelNum1_, axis=0) > self.thresh_UtypeLessDVoxelNum2_
        if TtypeCompareOuter:
            twoAreaCompare = np.sum(np.sum((mainArea[:, np.newaxis] -  secondArea) > self.threshold, axis=1) > self.thresh_TtypeMoreDVoxelNum1, axis=0) > self.thresh_TtypeMoreDVoxelNum2
        if TtypeCompareInner:
            twoAreaCompare = np.sum(np.sum((mainArea[:, np.newaxis] -  secondArea) > self.threshold, axis=1) > self.thresh_TtypeLessDVoxelNum1, axis=0) > self.thresh_TtypeLessDVoxelNum2
        if TtypeCompareOuter_:
            twoAreaCompare = np.sum(np.sum((mainArea[:, np.newaxis] -  secondArea) > self.threshold, axis=1) > self.thresh_TtypeMoreDVoxelNum1_, axis=0) > self.thresh_TtypeMoreDVoxelNum2_
        if TtypeCompareInner_:
            twoAreaCompare = np.sum(np.sum((mainArea[:, np.newaxis] -  secondArea) > self.threshold, axis=1) > self.thresh_TtypeLessDVoxelNum1_, axis=0) > self.thresh_TtypeLessDVoxelNum2_
        if CrossUp:
            twoAreaCompare = np.sum((mainArea[np.newaxis] -  secondArea) < (-self.threshold), axis=0) > self.thresh_CrossUpNum
        if CrossLeft:
            twoAreaCompare = np.all(np.sum((mainArea[:, np.newaxis] -  secondArea) < (-self.threshold), axis=1) > self.thresh_CrossLeftNum, axis=0)
        if CrossRight:
            twoAreaCompare = np.all(np.sum((mainArea[:, np.newaxis] -  secondArea) < (-self.threshold), axis=1) > self.thresh_CrossRightNum, axis=0)
        if CrossMiddle:
            twoAreaCompare = np.all(np.sum((mainArea[:, np.newaxis] -  secondArea) < (-self.threshold), axis=1) > self.thresh_CrossMiddleNum, axis=0)
        return twoAreaCompare
            
    def build_blocks(self, *points):
        return np.stack((points))

def constraint_Points_coordinate(point_list, lower=0, upper=255):
    '''
    constraint the points coordinate in the range of [lower, upper]
    and remove the duplicate points
    '''
    point_list = np.array(point_list)
    invalid_indices = np.where((point_list < lower) | (point_list > upper))[0]
    filtered_points = np.delete(point_list, invalid_indices, axis=0)
    filtered_points = np.unique(filtered_points, axis=0)
    return filtered_points.tolist()

def compute_12K_distance(fid, topk):
    fid = np.array(fid).reshape(1,3)
    return np.linalg.norm(np.array(topk)  - np.array(fid)[:, np.newaxis], axis=2).squeeze()

def turnarray_sum2one(array1d):
    sum_ = np.sum(array1d)
    array1d = array1d/sum_
    return array1d

def generate_random_unit_vector():

    phi = random.uniform(0, 2 * np.pi)
    costheta = random.uniform(-1, 1)
    theta = np.arccos(costheta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z])

def generate_coordinates(original_coordinate, num_coordinates=100, max_translation=5, min_translation=1, keeporiginpoint=True):
    if keeporiginpoint:
        coordinates = [original_coordinate.tolist()]
    else: 
        coordinates = []
        num_coordinates+=1
    for _ in range(num_coordinates-1):
        random_vector = generate_random_unit_vector()
        translation = random_vector * random.uniform(min_translation, max_translation)
        new_coordinate = np.rint(original_coordinate + translation).astype(np.int16).tolist()
        coordinates.append(new_coordinate)
    return coordinates

class AugPoints(object):

    def __init__(self, num_coordinates=10, max_translation=10, min_translation=1):
        self.num_coordinates = num_coordinates
        self.max_translation = max_translation 
        self.min_translation = min_translation 
        
    def __call__(self, ftpts):
        points_list = []

        for ftpt in ftpts:
    
            tanslate_coord = generate_coordinates(ftpt, self.num_coordinates, self.max_translation, self.min_translation)
            points_list += tanslate_coord
        return points_list
    
def tensor2numpy_squeeze0(tensor):
    return tensor.squeeze(0).numpy()