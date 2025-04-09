'''
get train datasets
get eval utils
'''
import random
import sys
import time
import numpy as np
import torch
import torchio as tio
from torch.utils.data import Dataset, DataLoader, ConcatDataset

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.util import *
from utils.PointDetection import Get_FeaturePoints

def get_train_datasets(args, config):
    mode = args.mode
    file_path_pkl = config['train_images_pkl']
    featpoint_path_re = config['train_featurePoints_re']

    file_paths = load_list_from_pkl(file_path_pkl)

    if featpoint_path_re is not None:
        if args.dataset == "sanbo":
            featpoint_paths = load_list_from_pkl(featpoint_path_re)
            featpoints = load_lists_from_pkls(featpoint_paths)
        else:
            featpoints_paths = get_pathlist_from_glob(featpoint_path_re)
            featpoints = load_lists_from_pkls(featpoints_paths)
    else:
        featpoints = None

    if mode == "train":
        end = time.time()
        train_dataset_list = []
        for i, sample in enumerate(MedImgDataset(file_paths, featpoints)):
            sub_dataset = CropDataset(sample, AugRotation=args.AugRotation, AugMask=args.AugMask, AugScale=args.AugScale, AugWeights=args.AugWeights, rotationParames=args.RotationParames)
            train_dataset_list.append(sub_dataset)
        trainCropsDataset = ConcatDataset(train_dataset_list)
        print("Build trainDataset cost time：", time.time()-end)
    elif mode == "test": 
        train_dataset_list = []
        for i, sample in enumerate(MedImgDataset(file_paths[0:1], featpoints[0:1])):
            sub_dataset = CropDataset(sample, AugRotation=args.AugRotation, AugMask=args.AugMask, AugScale=args.AugScale, AugWeights=args.AugWeights, rotationParames=args.RotationParames)
            train_dataset_list.append(sub_dataset)
        trainCropsDataset = ConcatDataset(train_dataset_list)
    elif mode == "eval":
        print("eval mode not need train dataset")
        trainCropsDataset = []

    return trainCropsDataset

def get_eval_datasets_onefold(args, config):
    
    mode = args.mode
    fidNum = args.fidNum
    pointselectRadius = args.ValidSelectRadius
    valid_path_pkl = config['valid_images_pkl_list'][0]
    valid_npy_paths = load_list_from_pkl(valid_path_pkl)
    validAfids_path_pkl = config['valid_afids_pkl_list'][0]
    valid_afids_paths = load_list_from_pkl(validAfids_path_pkl)
    valid_afidlists = load_lists_from_pkls(valid_afids_paths)
    valid_afidlists_np = np.array(valid_afidlists)

    template_config = config['template']
    if args.dataset == "sanbo":
        template_paths = load_list_from_pkl(template_config['template_save_re'])
        template_afid_paths = load_list_from_pkl(template_config['template_fids_re']) 
    else:
        template_paths = get_pathlist_from_glob(template_config['template_save_re']) + get_pathlist_from_glob(template_config['template_standardize_re'])
        template_afid_paths = get_pathlist_from_glob(template_config['template_fids_re']) +  get_pathlist_from_glob(template_config['template_afids_re'])
    template_afids = load_lists_from_pkls(template_afid_paths)

    template_dataloaders, meanTime = getInference_dataloaders(template_paths, template_afids, fidNum=fidNum)
    
    if args.dataset == "sanbo":
        fuseTemplate_dataloaders = None
    else:
        fuseTemplate_paths = load_list_from_pkl(config['valid_images_pkl_list'][1]) + load_list_from_pkl(config['valid_images_pkl_list'][2])
        fuseTemplate_afids_pkls = load_list_from_pkl(config['valid_afids_pkl_list'][1]) + load_list_from_pkl(config['valid_afids_pkl_list'][2])
        fuseTemplate_afidslists = load_lists_from_pkls(fuseTemplate_afids_pkls)
        fuseTemplate_dataloaders, meanTime = getInference_dataloaders(fuseTemplate_paths, fuseTemplate_afidslists, fidNum=fidNum)

    forSelectTemplateAfids = np.array(template_afids[config['template_ind']])
    if fidNum:
        valid_afidlists = valid_afidlists_np[:, :fidNum, :].tolist()
        forSelectTemplateAfids = forSelectTemplateAfids[:fidNum, :].tolist()

    if mode == "test" : 
        valid_Pointlists, screenMeanTime = Get_FeaturePoints(args, valid_npy_paths[:2])
        valid_Pointlists = validPoints_Select(forSelectTemplateAfids, valid_Pointlists, threshold=pointselectRadius)
        valid_dataloaders, validDataMeanTime = getInference_dataloaders(valid_npy_paths[0:2], valid_Pointlists[0:2], batchSize=args.batchSize_eval)
    else:
        valid_Pointlists, screenMeanTime = Get_FeaturePoints(args, valid_npy_paths)
        valid_Pointlists = validPoints_Select(forSelectTemplateAfids, valid_Pointlists, threshold=pointselectRadius)
        valid_dataloaders, validDataMeanTime = getInference_dataloaders(valid_npy_paths, valid_Pointlists, batchSize=args.batchSize_eval)
    

    valid_dict = {'valid_dataloaders': valid_dataloaders, 
                  "template_dataloaders": template_dataloaders, 
                  "fuseTemplate_dataloaders": fuseTemplate_dataloaders, 
                  'valid_MRI_paths': valid_npy_paths,
                  'valid_afidlists': valid_afidlists,
                  'valid_Pointlists':valid_Pointlists, 
                  'template_paths':template_paths, 
                  'template_fids': template_afids,
                  }
    return valid_dict

def getInference_dataloaders(filepaths, filepoints, batchSize=512, fidNum=None, fileType='path'):
    '''
    MRIDataset 支持输入输数据(array)或者数据路径(path)，使用fileType进行标识  
    '''
    dataloader_list = []
    end = time.time()
    if fidNum:
        batchSize = fidNum
    for i, sample in enumerate(MRIDatasetforInference(filepaths, filepoints, fileType)):
        sub_dataset = CropDatasetForInference(sample, fidNum=fidNum)
        sub_loader = DataLoader(sub_dataset, batch_size=batchSize, num_workers=4, pin_memory=True)
        dataloader_list.append(sub_loader)
    end = time.time() - end
    return dataloader_list, end/len(filepaths)


class MRIDatasetforInference(Dataset):
    def __init__(self, MRInames, Pointlists, fileType='path'):
        self.MRInames = MRInames
        self.Pointlists = Pointlists
        self.fileType = fileType
        
    def __len__(self):
        return len(self.MRInames)

    def __getitem__(self, idx):
        if self.fileType == 'path':
            M = np.load(self.MRInames[idx]) #MRI三维体素数据，存储为numpy.array
        elif self.fileType == 'array':
            M = self.MRInames[idx]
        Plist = self.Pointlists[idx]

        sample = {"MRI":M,  "MRI_pointlist":Plist}
       
        return sample

class CropDatasetForInference(Dataset):
    '''输入一个数据以及对应的特征点坐标，输出这个数据点的切块
        应用在编码器的推理阶段
    '''
    def __init__(self, MRIsample, fidNum=None, cropshape=32) -> None:
        # sample = {"MRI":M, "MRI_pointlist": Plist}
        self.MRI = MRIsample['MRI']
        self.pointslist = MRIsample['MRI_pointlist']
        self.cropshape = cropshape
        if fidNum:
            self.pointslist = self.pointslist[:fidNum]
        # self.pointsLabel = MRIsample['MRI_pointLabels']
        # self.afids_SpaceLabelRange = afids_SpaceLabelRange  # 6 (min3, max3)
        # self.filter_InSpaceRangePoints()
        
    def __len__(self):
        return len(self.pointslist)

    def pad2SpecifyShape(self, data, specifyShape):
        #对输入的数据进行填充到指定的形状
        diffs = specifyShape - np.array(data.shape)
        padsize = []
        for diff in np.ceil(diffs).astype(int):
            if diff%2 == 0:
                padsize.append([int(diff/2),int(diff/2)])
            else:
                padsize.append([int(diff/2),int(diff/2)+1])
        # print(f"类型：{type(np.pad(data, tuple(padsize), constant_values=(0,0)))}")
        return np.pad(data, tuple(padsize), constant_values=(0,0))
    
    def crop(self, m, p):
        # assert np.all(np.array(p) > -16)
        #计算切块起始终止位置，确保不越界
        cropsize = self.cropshape
        w = cropsize
        p[p < ( w / 2)] = w / 2
        p[p > (255 - w / 2) ] = 255 - w / 2

        start = np.round(np.maximum(np.array(p) - cropsize // 2, 0)).astype(np.int64)
        end = np.round(np.minimum(np.array(p) + cropsize // 2, m.shape)).astype(np.int64)
        block = m[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

        # #如果有越界的块，则将其填充至指定大小
        # if np.any(block.shape<(cropsize, cropsize, cropsize)):
        #     block = self.pad2SpecifyShape(block, (cropsize, cropsize, cropsize))
        
        return block[np.newaxis, :].astype(np.float32)

    def __getitem__(self, index) :
        sp = self.pointslist[index]      #当前特征点的坐标
        # sl = self.pointsLabel[index]
        sb = self.crop(self.MRI, np.array(sp, dtype=int).reshape(3)) # 特征点block
        return torch.tensor(sb)

class MedImgDataset(Dataset):
    def __init__(self, ImgPaths, Pointlists=None, SAVE=False, featurepoint_savepath=None, pointAug=False, transform=None):
        '''
        ImgPaths: img path list, a path is .npy file
        Pointlists: feature points list, a list of list, each list contains feature points of a sample
                    if Pointlists is None, detect feature points 
        '''
        self.ImgPaths = ImgPaths
        if Pointlists is not None:
            self.Pointlists = Pointlists
        else:
            self.Pointlists, _ = Get_FeaturePoints(self.ImgPaths, Aug=pointAug)
            if SAVE:
                for i, imgPath in enumerate(self.ImgPaths):
                    if os.path.isdir(featurepoint_savepath):
                        savepath = os.path.join(featurepoint_savepath, get_nameIn(imgPath), get_nameIn(imgPath)+"_featurePoints.pkl")
                        checkPath_mkdirs(savepath)

                        save_list_to_pkl(self.Pointlists[i], savepath)
                        print(f"{savepath}已保存,特征点个数{len(self.Pointlists[i])}")

                    else:
                        raise Exception('featurepoint_savepath is not a dir')
        self.transform = transform   
        
    def __len__(self):
        return len(self.ImgPaths)

    def __getitem__(self, idx):
        M = np.load(self.ImgPaths[idx]) #MRI三维体素数据，存储为numpy.array

        if self.transform is not None:
            for trans in self.transform:
                if trans is not None:
                    M = trans(M)

        Plist = self.Pointlists[idx]
        sample = {"imgArr": M, "pointList": Plist}
       
        return sample
    

class CropDataset(Dataset):
    # '每一个MRI的所有特征点切块作为一个完整的数据集，每次取得一个特征点在M和M_aug上对应位置的切块'
    # 在__getitem__中进行切块并返回，这样能够节省内存，
    def __init__(self, MRIsample, AugRotation=True, AugMask=False, AugScale=False, 
                 AugWeights=[1], crop_shape=32, maskPointNum=10, maskcrop_low=4, maskcrop_high=6, 
                 rotationParames=40, scaleRatioLower=0.7, scaleRatioUpper=1.3) -> None:
        self.sorceMRI = MRIsample['imgArr']
        self.sorce_pointslist = MRIsample['pointList']

        self.AugRotation = AugRotation
        self.AugMask = AugMask
        self.AugScale = AugScale

        # 切块大小
        self.crop_shape = crop_shape

        # aug params
        self.rotationParames = rotationParames
        self.scaleRatioLower = scaleRatioLower
        self.scaleRatioUpper = scaleRatioUpper
        self.maskPointNum = maskPointNum
        self.maskcrop_low = maskcrop_low
        self.maskcrop_high = maskcrop_high

        self.aug_MRIs = self.get_aug_Img()
        self.AugWeights = AugWeights

    def __len__(self):
        return len(self.sorce_pointslist)
    
    def set_random_params(self):
        '''each img generate one set of aug params'''
        rotation = ((-10, self.rotationParames), (-10, 10), (-10, 10))
        aug_params = {
                       'RotationX':random.uniform(*rotation[0]), 
                       'RotationY':random.uniform(*rotation[1]), 
                       'RotationZ':random.uniform(*rotation[2]), 
                       'ScaleRatio': random.uniform(self.scaleRatioLower, self.scaleRatioUpper)}
        return aug_params
    
    def get_aug_Img(self):
        ''' get aug Img array, Img pointlist'''

        # get aug params
        aug_params = self.set_random_params()
        # save array transformed
        aug_MRIs = {}
        # tio Image
        tio_MRI = tio.ScalarImage(tensor=torch.tensor(self.sorceMRI).unsqueeze(0))

            # rotation
        if self.AugRotation:
            Rotation, _ = transMatrix(1, 0, 0, 0, aug_params['RotationX'], aug_params['RotationY'], aug_params['RotationZ'])
            rotation_array, rotation_Pointlist, _ = ImageArray_3DTransformeWithPoint(self.sorceMRI, np.array(self.sorce_pointslist), Rotation, 1, _)
            aug_MRIs['Rotationed'] = rotation_array
            aug_MRIs['RotationedPointList'] = rotation_Pointlist.tolist()
            # scale
        if self.AugScale:
            tio_randomScale = tio.Resample(aug_params['ScaleRatio'])
            Scale_image = tio_randomScale(tio_MRI)
            Scale_array = tensor2numpy_squeeze0(Scale_image.data)
            Scale_pointlist = (np.array(self.sorce_pointslist) / aug_params['ScaleRatio']).tolist()
            aug_MRIs['Scaled'] = Scale_array
            aug_MRIs['ScaledPointList'] = Scale_pointlist
     
        return aug_MRIs
       
    def __getitem__(self, index) :
        sourcepoint = self.sorce_pointslist[index]      #sorce当前特征点的坐标
        sourcepoint = np.array(sourcepoint, dtype=int).reshape(3)

        sourceCrop = self.crop(self.sorceMRI, sourcepoint)#sorce的特征点block
        
        augLabels = self.AugWeights
        # 根据标志信息进行增广，整体增广、小块增广
        crops = {'sourceCrop': sourceCrop[np.newaxis, :]}
        augCrops = []

        if self.AugRotation:
            rotationed = self.aug_MRIs['Rotationed']
            rotationedPointList = self.aug_MRIs['RotationedPointList']
            rotationPoint = np.array(rotationedPointList[index])
            rotationCrop = self.crop(rotationed, rotationPoint)
            augCrops.append(rotationCrop[np.newaxis, :])

        if self.AugScale:
            Scaled = self.aug_MRIs['Scaled']
            ScaledPointList = self.aug_MRIs['ScaledPointList']
            ScalePoint = np.array(ScaledPointList[index])
            ScaleCrop = self.crop(Scaled, ScalePoint)
            augCrops.append(ScaleCrop[np.newaxis, :])

        if self.AugMask:
            nasmaskedCrop = sourceCrop * self.get_randomMask()
            augCrops.append(nasmaskedCrop[np.newaxis, :])
       
        crops['augCrops'] = np.array(augCrops)
        crops['augLabels'] = np.array(augLabels)
        crops['anchorCoords'] = sourcepoint
        
        return crops   
    
    def get_randomMask(self):
        random_points = torch.randint(0, self.crop_shape-1, size=(self.maskPointNum, 3))  # 生成400个随机点的坐标
        
        mask = torch.ones((self.crop_shape, self.crop_shape, self.crop_shape))
        
        for point in random_points:
            x, y, z = point
            # 计算掩码范围
            x_min = max(0, x - random.randint(self.maskcrop_low, self.maskcrop_high))
            x_max = min(self.crop_shape, x + random.randint(self.maskcrop_low, self.maskcrop_high))
            y_min = max(0, y - random.randint(self.maskcrop_low, self.maskcrop_high))
            y_max = min(self.crop_shape, y + random.randint(self.maskcrop_low, self.maskcrop_high))
            z_min = max(0, z - random.randint(self.maskcrop_low, self.maskcrop_high))
            z_max = min(self.crop_shape, z + random.randint(self.maskcrop_low, self.maskcrop_high))
            # 在掩码中将对应范围内的值设为1
            mask[x_min:x_max, y_min:y_max, z_min:z_max] = 0
        # print(len(mask[mask==0])/32**3)
        return mask.numpy()
        
    def pad2SpecifyShape(self, data, specifyShape):
        #对输入的数据进行填充到指定的形状
        diffs = specifyShape - np.array(data.shape)
        padsize = []
        for diff in np.ceil(diffs).astype(int):
            if diff%2 == 0:
                padsize.append([int(diff/2),int(diff/2)])
            else:
                padsize.append([int(diff/2),int(diff/2)+1])
        # print(f"类型：{type(np.pad(data, tuple(padsize), constant_values=(0,0)))}")
        return np.pad(data, tuple(padsize), constant_values=(0,0))
    
    def crop(self, m, p):
        '''在剪裁之前，首先将点的坐标放在有效切块范围内，这样导致在旋转之后超出范围的数据点不至于留白过多，
            另一方面这样会导致假阳性的出现'''
        p = np.array(p)
        w = self.crop_shape
        p[p < ( w / 2)] = w / 2
        p[p > (255 - w / 2) ] = 255 - w / 2
        #计算切块起始终止位置，确保不越界
        start = np.round(np.array(p) - np.array([self.crop_shape,self.crop_shape,self.crop_shape]) // 2).astype(np.int64)
        end = np.round(np.array(p) + np.array([self.crop_shape,self.crop_shape,self.crop_shape]) // 2).astype(np.int64)
        block = m[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

        # #如果有越界的块，则将其填充至指定大小
        if np.any(block.shape<(self.crop_shape, self.crop_shape, self.crop_shape)):
            block = self.pad2SpecifyShape(block, (self.crop_shape, self.crop_shape, self.crop_shape))
        
        return block.astype(np.float32)