import random
import sys
import time
import numpy as np
import torch
import torchio as tio
from torch.utils.data import Dataset, DataLoader, ConcatDataset

class MRIDatasetforInference(Dataset):
    def __init__(self, MRIs, Pointlists):

        self.MRIs = MRIs
        self.Pointlists = Pointlists
        
        
    def __len__(self):
        return len(self.MRIs)

    def __getitem__(self, idx):
        M = self.MRIs[idx]
        if isinstance(M, str):
            if M.endswith('.npy'):
                M = np.load(M) 
            elif M.endswith('.nii.gz'):
                M = tio.ScalarImage(M)
                M = M.data.squeeze().numpy()
        elif isinstance(M, np.ndarray):
            M = M

        Plist = self.Pointlists[idx]

        sample = {"MRI":M,  "MRI_pointlist":Plist}
        return sample
    
class CropDatasetForInference(Dataset):
    '''Input data and corresponding feature points, output cropped blocks
        Used in the inference stage of the encoder
    '''
    def __init__(self, MRIsample, fidNum=None, cropshape=32) -> None:
        # sample = {"MRI":M, "MRI_pointlist": Plist}
        self.MRI = MRIsample['MRI']
        self.pointslist = MRIsample['MRI_pointlist']
        self.cropshape = cropshape
        if fidNum:
            self.pointslist = self.pointslist[:fidNum]

        
    def __len__(self):
        return len(self.pointslist)

    def pad2SpecifyShape(self, data, specifyShape):
        # Pad input data to specified shape
        diffs = specifyShape - np.array(data.shape)
        padsize = []
        for diff in np.ceil(diffs).astype(int):
            if diff%2 == 0:
                padsize.append([int(diff/2),int(diff/2)])
            else:
                padsize.append([int(diff/2),int(diff/2)+1])
        return np.pad(data, tuple(padsize), constant_values=(0,0))
    
    def crop(self, m, p):
        # Calculate crop start and end positions to ensure no out-of-bounds
        cropsize = self.cropshape
        w = cropsize
        p[p < ( w / 2)] = w / 2
        p[p > (255 - w / 2) ] = 255 - w / 2

        start = np.round(np.maximum(np.array(p) - cropsize // 2, 0)).astype(np.int64)
        end = np.round(np.minimum(np.array(p) + cropsize // 2, m.shape)).astype(np.int64)
        block = m[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

        return block[np.newaxis, :].astype(np.float32)

    def __getitem__(self, index) :
        sp = self.pointslist[index]      
        # sl = self.pointsLabel[index]
        sb = self.crop(self.MRI, np.array(sp, dtype=int).reshape(3)) 
        return torch.tensor(sb)
    

def get_dense_fp_dl(args, templateAfids, validArray):
    '''
    After registration, template afid is near the valid afid, 
    use transed template afid to get the dense valid afid
    '''
   
    valid_selectPoints = validPoints_denseSample(templateAfids, validSelectRadius=args.ValidSelectRadiusReg)
    validSelectedDL = get_Inference_dl_one_arr(args, validArray, valid_selectPoints[0])
    # validSelectedDL, _ = getInference_dataloaders([validArray], valid_selectPoints, fileType='array')
    return validSelectedDL, valid_selectPoints[0]


def validPoints_denseSample(templateafids, validSampleNum=1, margin=1, validSelectRadius=25):
    '''Densely sample around template anatomical landmarks to estimate the region of interest
    1. Sample around template anatomical landmarks with 25 pixel radius and 2 pixel interval
    '''
    validDensePoints = []
    templateafids = np.array(templateafids).astype(int)
    # Loop through all template anatomical landmarks
    for templateafid in templateafids:
        # Calculate min and max bounds for each dimension
        min = templateafid-validSelectRadius
        min[min<0] = 0
        max = templateafid+validSelectRadius
        max[max>255]=255
        points = [[i,j,k] for i in range(min[0], max[0], margin) for j in range(min[1], max[1], margin) for k in range(min[2], max[2], margin)]
        validDensePoints += points
    # Remove duplicate points
    validDensePoints = np.unique(validDensePoints, axis=0).tolist()
    # Create candidate anatomical landmarks for each validation data
    valid_pointLists = [validDensePoints for _ in range(validSampleNum)]
    return valid_pointLists

def get_Inference_dl_one_arr(args, Img_arr, feature_points):
    ds = CropDatasetForInference({"MRI":Img_arr, "MRI_pointlist":feature_points}, cropshape=args.cropSize)
    dl = DataLoader(ds, 
                    batch_size=args.batchSize_eval, 
                    shuffle=False, 
                    num_workers=args.num_workers, 
                    pin_memory=args.pin_memory)
    return dl

def DL_infer(args, model, dataloader):
    # model = model.to(args.device)
    fp_embeds = torch.tensor([])
    for crops in dataloader:
        torch.cuda.empty_cache()
        fp_embed = model(crops.to(args.device)).cpu().detach()  # bs * 512
        fp_embeds = torch.cat([fp_embeds, fp_embed], dim=0)  # len(dl) * bs * 512
    return fp_embeds

def getInference_dataloaders(args, filepaths, filepoints, fidNum=None):
    '''
    MRIDataset supports input data (array) or file paths (path), identified by fileType
    '''
    dataloader_list = []
    batchSize = args.batchSize_eval
    if fidNum:
        batchSize = fidNum
    for i, sample in enumerate(MRIDatasetforInference(filepaths, filepoints)):
        sub_dataset = CropDatasetForInference(sample, fidNum=fidNum, cropshape=args.cropSize)
        sub_loader = DataLoader(sub_dataset, 
                                batch_size=batchSize,
                                shuffle=False, 
                                num_workers=args.num_workers, 
                                pin_memory=args.pin_memory)
        dataloader_list.append(sub_loader)

    return dataloader_list