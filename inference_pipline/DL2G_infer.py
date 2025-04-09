'''
input one image, output the fiducial points

Predefine: 
1. voxel sapce: 1mm*1mm*1mm
2. image size: 256*256*256
3. fiducial points: n points
4. template images: m images, type: list[ndarray], shape: 256*256*256
5. experiment setting: 
    DL_only(Deep Local only) must be True,
    GC(Geometric Constraint), 
    GC_wo_Reg(Geometric Constraint without Registration),
    GC_wo_Reg_Dir(Geometric Constraint without Registration and Direction)

Preprocess: voxel intensity[0, 256], center of mass, resize to 256*256*256  
note that the center of mass need to save translate information to recover the image to source space


Inference: load the model, input the image, detect the feature points
           DL: 
               extract the feature points crop feature embedding on the input image, 
               extract the fiducial points crop feature embedding on the template images, fuse the features,
               compare cosine similarity to get the fiducial points candidates

           if GC
               Distance and Direction Constraint candidates, get the matched fiducial points (GC_wo_Reg)
               Registration the template images to the input image,
               Get the dense feature points on the input image use the transformed template fiducial points,
               Compare the feature points and template fiducial embedding cosine similarity to get the fiducial points candidates
               Distance and Direction Constraint new candidates, get the matched fiducial points(GC)
               if center of mass：
                    translate the fiducial points to the source space

            return the fiducial points in the source space
'''

import sys
import time
import torch
import argparse
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset


sys.path.append(os.path.dirname(__file__))
from utils.util import *
import utils.logger as logger
from utils.PointDetection import Get_featurePoints_one_image as pd_one_img
from utils.PointDetection import Get_FeaturePoints as pd
from utils.PointDetection import InferPoints_localSelect 
from utils.GeoConstraint import GC_with_reg
import model.Resnet as Resnet
import model.MoCo_builder as MoCo_builder
from dataset.dataload import CropDatasetForInference, getInference_dataloaders, get_Inference_dl_one_arr, DL_infer, get_dense_fp_dl

def Predefine(args, config):
    '''
    1. voxel sapce: 1mm*1mm*1mm
    2. image size: 256*256*256
    3. fiducial points: n points
    4. template images: m images, type: list[ndarray], shape: 256*256*256
    '''

    # template images
    if args.dataset == 'sanbo':
        template_paths = load_list_from_pkl(config['template']['template_save_re'])
        template_fids_paths = load_list_from_pkl(config['template']['template_fids_re'])
    else:
        template_fids_paths = get_pathlist_from_glob(config['template']['template_fids_re'])
        template_paths = get_pathlist_from_glob(config['template']['template_save_re'])
    template_fids_lists = load_lists_from_pkls(template_fids_paths)

    if args.dataset == 'sanbo':
        few_shot_paths = template_paths
        few_shot_fids_lists = template_fids_lists
    else:
        # source images
        source_paths = load_list_from_pkl(config['oasis']['valid_images_pkl_list'][1])
        source_fids_paths = load_list_from_pkl(config['oasis']['valid_afids_pkl_list'][1])
        source_fids_lists = load_lists_from_pkls(source_fids_paths)

        #few shot template images
        few_shot_paths = [template_paths[config['template']['template_ind']], source_paths[0]]
        few_shot_fids_lists = [template_fids_lists[config['template']['template_ind']], source_fids_lists[0]]
        # last three point
        few_shot_fids_lists = [few_shot_fids_list[-3:] for few_shot_fids_list in few_shot_fids_lists]


        few_shot_paths = [few_shot_paths[-1]]
        few_shot_fids_lists = [few_shot_fids_lists[-1]]
    print("template images: ", len(few_shot_paths))
    print("fiducial points: ", len(few_shot_fids_lists[0]))

    return few_shot_paths, few_shot_fids_lists


def Preprocess(image_path, config):
    '''
    Preprocess: voxel intensity[0, 256], center of mass, resize to 256*256*256
    note that the center of mass need to save translate information to recover the image to source space
    image_path: str with postfix .nii.gz
    return image: ndarray, shape: 256*256*256
    '''

    # need to record the direction information of img
    # image is processed, this part does not need to be implemented
    pass

def get_template_embeds(args, model, template_paths, template_fids_lists):
    template_afids_dls = getInference_dataloaders(args, template_paths, template_fids_lists, fidNum=args.fidNum)
    template_fids_embedSUM = 0
    # get few shot embeddings
    for i, dl_tempalte in enumerate(template_afids_dls):
        crops = next(iter(dl_tempalte)).to(args.device)
        template_fids_embed = model(crops).cpu().detach()  # afidsNum * encoderDim
        
        template_fids_embedSUM += template_fids_embed
    few_shot_embeds = template_fids_embedSUM / len(template_afids_dls)

    # get template image
    template_img = template_paths[args.template_ind]
    template_fids_list = template_fids_lists[args.template_ind]
    if isinstance(template_img, str):
        if template_img.endswith('.npy'):
            template_img = np.load(template_img) 
        elif template_img.endswith('.nii.gz'):
            template_img = tio.ScalarImage(template_img)
            template_img = template_img.data.squeeze().numpy()
    elif isinstance(template_img, np.ndarray):
        template_img = template_img
    template_utils = {
        'template_img': template_img,
        'template_fids_list': template_fids_list,
        'template_fids_embed': few_shot_embeds,
    }
    return template_utils
def get_model_encoder(args):

    print(f"build model:{args.arch}")
    if args.arch == 'resnet18':
        res = Resnet.resnet18
    elif args.arch == 'resnet10':
        res = Resnet.resnet10
    model = MoCo_builder.MoCo(res, 
                              dim=args.moco_dim, #"feature dimension (default: 128)"
                              m=args.moco_m, #"moco momentum of updating key encoder (default: 0.999)"
                              K=args.moco_k, #"queue size; number of negative keys (default: 65536)"
                              T=args.moco_t,  #softmax temperature (default: 0.07)
                              mlp=args.mlp 
                              )
    print(f"load checkpoint from {args.resume}")
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['state_dict'])
    encoder = model.get_submodule('encoder_q')
    return encoder.eval()


def DL_GC(args, image, model, template_img, template_fids, template_fids_embeds):
    '''
    Inference: load the model, input the image, detect the feature points
    return fiducial points in the source space

    image: ndarray, shape: 256*256*256
    '''
    cos = nn.CosineSimilarity(dim=2)
    # feature point detection
    end = time.time()
    feature_points = pd_one_img(args, image)
    feature_points = InferPoints_localSelect(template_fids, [feature_points], args.ValidSelectRadius)[0]
    # build the feature points dataset
    fp_crop_dataloader = get_Inference_dl_one_arr(args, image, feature_points)
    PD_time = time.time() - end
    print(f"PD time: {PD_time}")
    # extract the feature embedding
    end = time.time()
    fp_embeds = DL_infer(args, model, fp_crop_dataloader)
    
    # DL: compare the cosine similarity
    afids_fp_logits = cos(template_fids_embeds.unsqueeze(1), fp_embeds)
    afid_SpaceLabel_Restrain_matrix = LocalSearch(template_fids, feature_points, threshold=args.ValidSelectRadius)  # afidNum * pointsNum
    afids_fp_logits = afids_fp_logits * afid_SpaceLabel_Restrain_matrix

    afids_tpok_values, afids_tpok_indexs = torch.topk(afids_fp_logits, args.CandidateNum, dim=1)
    feature_points = torch.tensor(feature_points)
    afids_topk_points = feature_points[afids_tpok_indexs]  # afidNum , k(topk), 3 
    # result['DL'] = afids_topk_points[:, 0].numpy().tolist()
    DL_landmarks = afids_topk_points[:, 0].numpy().tolist()
    DL_time = time.time() - end
    print(f"DL time: {DL_time}")
    # if GC
    end = time.time()
 
    if args.GC :
        # Distance and Direction Constraint candidates, get the matched fiducial points, registration the template images to the input image
        lds_gc_wo_reg, topkindex_lds_gc_wo_reg, Reged_TemplateArray, Reged_template_afids = GC_with_reg(template_img, image, template_fids, afids_topk_points)  # afidNum,  3
        
        # Get the dense feature points on the input image use the transformed template fiducial points
        RegDenseDL, DenseFp = get_dense_fp_dl(args, Reged_template_afids, image)
        DenseFp_embeds = DL_infer(args, model, RegDenseDL) 
        
        # Compare the feature points and template fiducial embedding cosine similarity to get the fiducial points candidates
        afids_DenseFp_logits = cos(template_fids_embeds.unsqueeze(1), DenseFp_embeds[0])   # afidNum * RegSelectPointNum
        
        Logits_localSearch_matrix = LocalSearch(Reged_template_afids, DenseFp, threshold=args.ValidSelectRadiusReg)  # afidNum * pointsNum
        afids_DenseFp_logits = afids_DenseFp_logits * Logits_localSearch_matrix
        afids_topk_Densefp_values, afids_tpok_DenseFpIndexs = torch.topk(afids_DenseFp_logits, args.CandidateNum, dim=1)
        DenseFp = torch.tensor(DenseFp)
        afids_topk_DenseFps = DenseFp[afids_tpok_DenseFpIndexs]
        
        # Distance and Direction Constraint new candidates, get the matched fiducial points
        lds_gc, topkindex_lds_gc, _, _ = GC_with_reg(Reged_TemplateArray, image, Reged_template_afids, afids_topk_DenseFps)  # afidNum,  3
        GC_landmarks = lds_gc

        # afids_RegPostReg_Dist = torch.norm((torch.tensor(validMRI_golds) - afids_RegRegPostReg).float(), dim=1) # 3
        # afids_PostFinalMReg_Dist = torch.norm((torch.tensor(validMRI_golds) - lds_gc).float(), dim=1) # 3
    GC_time = time.time() - end
    print(f"GC time: {GC_time}")
    # total_time = PD_time + DL_time + GC_time
    total_time = {}
    total_time['PD'] = PD_time
    total_time['DL'] = DL_time
    total_time['GC'] = GC_time
    return DL_landmarks, GC_landmarks, total_time

def infer_one_image(args, config, image):
    # args = parase()
    if args.gpu is not None:
        args.device = torch.device(f'cuda:{args.gpu}')
    else:
        args.device = torch.device('cpu')

    # config = load_config_from_yaml(args.config)
    # args.fidNum = config['template']['fidNum']
    # args.template_ind = config['template']['template_ind']

    # model
    model = get_model_encoder(args)
    model = model.to(args.device)
    # template
    template_paths, template_fids_lists = Predefine(args, config)
    template_utils = get_template_embeds(args, model, template_paths, template_fids_lists)
    # inference
    DL_landmarks, GC_landmarks, time = DL_GC(args, 
                                             image, 
                                             model, 
                                             template_img=template_utils['template_img'],
                                             template_fids=template_utils['template_fids_list'],
                                             template_fids_embeds=template_utils['template_fids_embed'])
    
    return DL_landmarks, GC_landmarks, time

def infer_images(args, config, images, gts_afids=None):

    GC_landmarks = []
    DL_landmarks = []
    times = 0
    for image in images:
        DL_landmark, GC_landmark, time = infer_one_image(args, config, image)
        DL_landmarks.append(DL_landmark)
        GC_landmarks.append(GC_landmark)

        times += time['PD'] + time['DL'] + time['GC']

    print(f"DL2G inference time: {times}, mean time: {times/len(images)}")
    if gts_afids is not None:
        mre_meter = AverageMeter("MRE", ":6.2f")
        sdr_meter = AverageMeter("SDR", ":6.2f")
        progress = ProgressMeter(len(DL_landmarks), [mre_meter, sdr_meter], prefix=f"DL {len(gts_afids[0])} point results:")
        # calculate the mre and std
        for i, (landmark, gt_afids) in enumerate(zip(DL_landmarks, gts_afids)):
            landmark = torch.tensor(landmark)
            gt_afids = torch.tensor(gt_afids)
            mre = torch.norm((landmark - gt_afids).float(), dim=1)
            sdr = compute_SDR(mre)

            mre_meter.update(np.array(mre).round(5))
            sdr_meter.update(sdr)
            progress.display(i+1)
        
        mre_meter = AverageMeter("MRE", ":6.2f")
        sdr_meter = AverageMeter("SDR", ":6.2f")
        progress = ProgressMeter(len(gts_afids[0]), [mre_meter, sdr_meter], prefix=f"GC {len(gts_afids[0])} point results:")
        # calculate the mre and std
        for i, (landmark, gt_afids) in enumerate(zip(GC_landmarks, gts_afids)):
            landmark = torch.tensor(landmark)
            gt_afids = torch.tensor(gt_afids)
            mre = torch.norm((landmark - gt_afids).float(), dim=1)
            sdr = compute_SDR(mre)

            mre_meter.update(np.array(mre).round(5))
            sdr_meter.update(sdr)
            progress.display(i+1)
def main():
    args = parase()
    dataset = args.dataset
    config = load_config_from_yaml(args.config)
    config = config[dataset]
    args.constMean = config['norm'][0]
    print(f"dataset mean: {args.constMean}")
    # if dataset == 'sanbo':
    #     args.preScreenSize = config['preScreenSize']
    #     args.fidNum = config['fidNum']
    #     args.augnum = config['augnum']
    for key in config:
        if hasattr(args, key):
            setattr(args, key, config[key])

    image_paths = load_list_from_pkl(config["valid_images_pkl_list"][0])
    gt_afids_path = load_list_from_pkl(config["valid_afids_pkl_list"][0])
    images = [np.load(path) for path in image_paths]
    # gt_afids = [load_list_from_pkl(path) for path in gt_afids_path]
    gt_afids = [load_list_from_pkl(path)[-3:] for path in gt_afids_path]
    args.fidNum = config['fidNum']
    print(f"detect {args.fidNum} fiducial points")
    args.template_ind = config['template_ind']
    print(f"use index {args.template_ind} as template")
    if args.log_path is not None:
        log_path = args.log_path
        checkPath_mkdirs(log_path)
        logger.set_log(f"dataset{dataset}_few_shot_1s1p_fidnum{args.fidNum}_ValidSelectRadius{args.ValidSelectRadius}_ValidSelectRadiusReg{args.ValidSelectRadiusReg}", log_path)

    infer_images(args, config, images, gt_afids)



def parase():
    parser = argparse.ArgumentParser(description="DL2G Landmark detection")
    parser.add_argument("--config", type=str, default="/home/wangrui/code/autoFidDetection/DL2G/inference_pipline/config.yaml", help="config file")
    parser.add_argument("--gpu", default=5, type=int, help="GPU id to use.")
    parser.add_argument("--cropSize", default=32, type=int, help="crop size")

    # model
    parser.add_argument("-a", "--arch", metavar="ARCH", default="resnet10", help="base_model")
    parser.add_argument("--moco-dim", default=512, type=int, help="feature dimension (default: 128)")
    parser.add_argument("--moco-k", default=65536, type=int, help="queue size; number of negative keys (default: 65536)")
    parser.add_argument("--moco-m", default=0.999, type=float, help="moco momentum of updating key encoder (default: 0.999)")
    parser.add_argument("--moco-t", default=0.07, type=float, help="softmax temperature (default: 0.07)")
    # options for moco v2
    parser.add_argument("--mlp", action="store_true", default=True, help="use mlp head")
    parser.add_argument("--resume", default="/data/wangrui/expResult/autoFidDetection/DL2G/sanbo/checkpoints/contrast/train/aug3_weights[0.33, 0.34, 0.33]/trainR_30_ValidR_True_30/AugRotation:True_AugMask:True_AugScale:True/resnet10_512_lr0.001_mlp_True_checkpoint_0000.pth.tar", type=str, metavar="PATH", help="path to latest checkpoint (default: none)")

    # inference
    parser.add_argument("--batchSize-eval", default=3000, type=int, help="batch size for eval")
    parser.add_argument("--dataset", default='sanbo', type=str, help="batch size for eval")
    parser.add_argument("--CandidateNum", default=20, type=int, help="the number of candidates")
    # parser.add_argument("--GC_mode", choices=['GC', 'GC_wo_Reg', 'GC_wo_Reg_Dir'], default='GC', help="Geometric Constraint mode")
    parser.add_argument("--GC", action="store_true", default=True, help="Geometric Constraint mode")
    parser.add_argument("--num_workers", default=0, type=int, help="number of workers")
    parser.add_argument("--pin_memory", action="store_true", default=False, help="pin memory")
    parser.add_argument("--ValidSelectRadius", default=30, type=int, help="the radius of valid select")
    parser.add_argument("--ValidSelectRadiusReg", default=10, type=int, help="the radius of valid select dense")
    
    # log
    parser.add_argument("--log_path", default="/data/wangrui/expResult/autoFidDetection/DL2G/inference_pipline/Logs/", type=str, help="log path")
    # feature point detection
    parser.add_argument("--screenAug", action="store_true", default=True, help="prescreen parameter")
    parser.add_argument("--resampleScale", default=4, type=int, help="prescreen parameter")
    parser.add_argument("--augnum", default=6, type=int, help="prescreen parameter")
    parser.add_argument("--mintranslate", default=1, type=int, help="prescreen parameter")
    parser.add_argument("--maxtranslate", default=10, type=int, help="prescreen parameter")
    parser.add_argument("--preScreenSize", default=1, type=int, help="prescreen parameter")

    parser.add_argument("--SaddleFace", action="store_true", default=False, help="prescreen diff type para")
    parser.add_argument("--SaddleAngle", action="store_true", default=False, help="prescreen diff type para")
    parser.add_argument("--SaddleArris", action="store_true", default=False, help="prescreen diff type para")
    parser.add_argument("--CornerT", action="store_true", default=False, help="prescreen diff type para")
    parser.add_argument("--CornerU", action="store_true", default=False, help="prescreen diff type para")
    parser.add_argument("--CornerTMinus", action="store_true", default=True, help="prescreen diff type para")
    parser.add_argument("--CornerUMinus", action="store_true", default=True, help="prescreen diff type para")
    parser.add_argument("--Cross", action="store_true", default=True, help="prescreen diff type para")

    parser.add_argument("--thresh_FaceDVoxelNum1", default=0, type=int, help="prescreen diff type para")
    parser.add_argument("--thresh_FaceDVoxelNum2", default=3, type=int, help="prescreen diff type para")
    parser.add_argument("--thresh_AngleDVoxelNum1", default=0, type=int, help="prescreen diff type para")
    parser.add_argument("--thresh_AngleDVoxelNum2", default=3, type=int, help="prescreen diff type para")
    parser.add_argument("--thresh_ArrisDVoxelNum1", default=0, type=int, help="prescreen diff type para")
    parser.add_argument("--thresh_ArrisDVoxelNum2", default=2, type=int, help="prescreen diff type para")
    parser.add_argument("--thresh_UtypeMoreDVoxelNum1", default=0, type=int, help="prescreen diff type para")
    parser.add_argument("--thresh_UtypeMoreDVoxelNum2", default=17, type=int, help="prescreen diff type para")
    parser.add_argument("--thresh_UtypeLessDVoxelNum1", default=17, type=int, help="prescreen diff type para")
    parser.add_argument("--thresh_UtypeLessDVoxelNum2", default=0, type=int, help="prescreen diff type para")
    parser.add_argument("--thresh_UtypeMoreDVoxelNum1_", default=0, type=int, help="prescreen diff type para")
    parser.add_argument("--thresh_UtypeMoreDVoxelNum2_", default=19, type=int, help="prescreen diff type para")
    parser.add_argument("--thresh_UtypeLessDVoxelNum1_", default=19, type=int, help="prescreen diff type para")
    parser.add_argument("--thresh_UtypeLessDVoxelNum2_", default=0, type=int, help="prescreen diff type para")
    parser.add_argument("--thresh_TtypeMoreDVoxelNum1", default=0, type=int, help="prescreen diff type para")
    parser.add_argument("--thresh_TtypeMoreDVoxelNum2", default=14, type=int, help="prescreen diff type para")
    parser.add_argument("--thresh_TtypeLessDVoxelNum1", default=0, type=int, help="prescreen diff type para")
    parser.add_argument("--thresh_TtypeLessDVoxelNum2", default=12, type=int, help="prescreen diff type para")
    parser.add_argument("--thresh_TtypeMoreDVoxelNum1_", default=2, type=int, help="prescreen diff type para")
    parser.add_argument("--thresh_TtypeMoreDVoxelNum2_", default=14, type=int, help="prescreen diff type para")
    parser.add_argument("--thresh_TtypeLessDVoxelNum1_", default=2, type=int, help="prescreen diff type para")
    parser.add_argument("--thresh_TtypeLessDVoxelNum2_", default=12, type=int, help="prescreen diff type para")
    parser.add_argument("--thresh_CrossUpNum", default=0, type=int, help="prescreen diff type para")
    parser.add_argument("--thresh_CrossLeftNum", default=0, type=int, help="prescreen diff type para")
    parser.add_argument("--thresh_CrossRightNum", default=1, type=int, help="prescreen diff type para")
    parser.add_argument("--thresh_CrossMiddleNum", default=1, type=int, help="prescreen diff type para")

    return parser.parse_args()


if __name__ == "__main__":
    main()