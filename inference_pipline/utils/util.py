import glob
import json
import os
import pickle
import warnings
import numpy as np
import torch
import yaml
import SimpleITK as sitk
import torchio as tio



# inference utils
def LocalSearch(template_fidcoordinates, validMRIs_Pointlists, threshold=30):
    '''
    compute the distance between each afid and each point, if the distance is less than threshold, the value is 1, otherwise 0 
    output: afidNum * pointsNum
    '''
    pointsNum = len(validMRIs_Pointlists)
    afidNum = len(template_fidcoordinates)
    template_Coordinates = torch.tensor(template_fidcoordinates)
    validMRI_PointCoordinates = torch.tensor(validMRIs_Pointlists).T
    # template_labels = torch.tensor(template_labels)
    # validMRI_labels = torch.tensor(validMRI_labels)
    template_Coordinates = template_Coordinates.reshape(-1) # (afidsNum*3)
    boradcast_template_Coordinates = template_Coordinates.repeat(pointsNum, 1).T  # （3*afidsNum）* pointsNum
    
    boradcast_queue_Coordinates = validMRI_PointCoordinates.repeat(afidNum, 1) # （3*afidsNum）* pointsNum

    MinusCoordinates = boradcast_queue_Coordinates - boradcast_template_Coordinates
    afid_SpaceLabelRestrain_indexMatrix = torch.norm(MinusCoordinates.reshape(-1, 3, pointsNum), dim=1) <= threshold# afidsNum * pointsNum


    # restrain_matrix = torch.zeros((afidNum, pointsNum))
    restrain_matrix = torch.ones((afidNum, pointsNum)) * 0.01  # 防止在验证时没有被抑制的点不足20这样需要其他被抑制的点来补齐，但是因为被抑制的点都一样，会从头开始选择，这样会产生较大的偏差
    restrain_matrix[afid_SpaceLabelRestrain_indexMatrix] = 1
    return restrain_matrix

def compute_SDR(dis):
    '''sdrsuccessful detection rates
    '''
    sdr_result = []
    # 循环计算：
    for i in [1, 2, 3, 4, 5]:
        i_sdr = len(dis[dis<i]) / len(dis)
        sdr_result.append(i_sdr)
    return np.array(sdr_result).round(4)

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()
        

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.std = 0
        self.vals = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(val)
        self.std = np.std(self.vals, axis=0)


    def __str__(self):
        fmtstr = "\n\t{name} :\n\t\tval: {val}({val_mean})\n\t\tavg: {avg}({avg_mean})\n\t\tstd: {std}({std_mean})"
        return fmtstr.format(name=self.name, val=self.val, val_mean=np.mean(self.val), avg=self.avg, avg_mean=np.mean(self.avg), std=self.std, std_mean=np.mean(self.std))


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"





# path utils
def load_config_from_yaml(file, name=None):
    with open(file, 'r') as file:  
        data = yaml.safe_load(file)  
    if name is not None and name in data:
        return data[name]
    return data

def get_pathlist_from_glob(repath):
    return sorted(glob.glob(repath))

def load_lists_from_pkls(pkl_paths):
    return [load_list_from_pkl(path) for path in pkl_paths]

def load_list_from_pkl(pklname):
    with open(pklname, 'rb') as file:
        data = pickle.load(file)
    return data

def get_nameIn(path, position=-1):
    p, n = os.path.split(path)
    if position==0:
        return n.split('.')[0]   
    return p.split('/')[position]   

def checkPath_mkdirs(path):
    directory = os.path.dirname(path)  
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)




