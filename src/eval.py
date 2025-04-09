'''eval contain all exp setting
few shot learning: Ap, 1s, Ap1s, Ap10s, ApAs
albation exp: local search, multi aug, geometric distance, geometric direction
'''
import os
import sys
import time
import torch
import numpy as np
import torch.nn as nn
sys.path.append(os.path.dirname(__file__))

from utils.misc import AverageMeter_Eval, ProgressMeter
from utils.util import get_nameIn
from datasets.dataload import getInference_dataloaders
from GeoConstraint import *


def eval_one_fold(model, eval_utils, args, config):
    fidNum = args.fidNum
    template_ind = config['template_ind']
    fuse_template_list = config['fuse_public_index']
    if args.gpu is not None:
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    model.eval()

    # get template features
    template_dataloaders = eval_utils['template_dataloaders']
    template_fids = eval_utils['template_fids']
    fuseTemplate_dataloaders = eval_utils['fuseTemplate_dataloaders']
    # template_fids_WithFuse = eval_utils['template_fids_WithFuse']
    if args.dataset == 'sanbo':
        templates_features ,template_fids_WithFuse = get_template_Cropfeature_Coordinates_sanbo(template_dataloaders, template_fids, fuseTemplate_dataloaders, model, fidNum=fidNum, fuse_template_list=fuse_template_list, template_ind=template_ind, device=device)
    else:
        templates_features ,template_fids_WithFuse = get_template_Cropfeature_Coordinates(template_dataloaders, template_fids, fuseTemplate_dataloaders, model, fidNum=fidNum, fuse_template_list=fuse_template_list, template_ind=template_ind, device=device)

    # get valid features
    valid_dataloaders = eval_utils['valid_dataloaders']
    print("Valid feature extraction...")
    validMRIs_features, meanTimesfeatureExtract = get_ValidMRIPoints_features(valid_dataloaders, model, device=device)    
    print(f"feature extraction mean time: {meanTimesfeatureExtract}s")

    # compute eval results
    valid_Pointlists = eval_utils['valid_Pointlists']
    valid_afidlists = eval_utils['valid_afidlists']
    template_paths = eval_utils['template_paths']
    valid_MRI_paths = eval_utils['valid_MRI_paths']

    template_paths.append(template_paths[template_ind])
    fid_error(args, model, templates_features, template_fids_WithFuse, validMRIs_features, valid_Pointlists, valid_afidlists, template_paths, valid_MRI_paths)

def fid_error(args, model, templates_features, template_fids_WithFuse, validMRIs_features, validMRIs_Pointlists, validMRIs_golds, template_paths, valid_MRI_paths):
    fusetemplate_paths, meter1, meter20, meterpostMRegfinal, meterRegPostReg, meterpostMRegwoDisfinal, meterRegPostRegwoDis, metersSDR_top1, metersSDR_postMReg, metersSDR_RegPostReg, metersSDR_postMRegwoDis, metersSDR_RegPostRegwoDis, metersTimes= get_AverageMeter_progressMeter(args, template_paths[:3])
    progress1 = ProgressMeter(len(validMRIs_features), meter1, prefix="top1 MRE:")
    progress20 = ProgressMeter(len(validMRIs_features), meter20, prefix="top20 MRE:")
    progressPostMRegFinal = ProgressMeter(len(validMRIs_features), meterpostMRegfinal, prefix="GC without Registration MRE:")
    progressRegPostReg = ProgressMeter(len(validMRIs_features), meterRegPostReg, prefix="Geometric Constraint(GC) MRE:")
    progressPostMRegwoDisFinal = ProgressMeter(len(validMRIs_features), meterpostMRegwoDisfinal, prefix="GC without Registration & Direction MRE:")
    progressRegPostRegwoDis = ProgressMeter(len(validMRIs_features), meterRegPostRegwoDis, prefix="GC without Direction MRE:")
    progressSDR_top1 = ProgressMeter(len(validMRIs_features), metersSDR_top1, prefix="top1 SDR:")
    progressSDR_postMReg = ProgressMeter(len(validMRIs_features), metersSDR_postMReg, prefix="GC without Registration SDR:")
    progressSDR_RegPostReg = ProgressMeter(len(validMRIs_features), metersSDR_RegPostReg, prefix="Geometric Constraint(GC) SDR:")
    progressSDR_postMRegwoDis = ProgressMeter(len(validMRIs_features), metersSDR_postMRegwoDis, prefix="GC without Registration & Direction SDR:")
    progressSDR_RegPostRegwoDis = ProgressMeter(len(validMRIs_features), metersSDR_RegPostRegwoDis, prefix="GC without Direction SDR:")
    progressTimes = ProgressMeter(len(validMRIs_features), metersTimes, prefix="Cost Time:")
    cos = nn.CosineSimilarity(dim=2)

    if args.gpu is not None:
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')

    for i, (valid_MRI_path, validMRI_features, validMRI_Pointlists, validMRI_golds) in enumerate(zip(valid_MRI_paths, validMRIs_features, validMRIs_Pointlists, validMRIs_golds)):
        topk_onesubject = []
        validMRI_Pointlists = np.array(validMRI_Pointlists)
        validMRI_array = np.load(valid_MRI_path)
        for j, (template_path, template_features, template_fidcoordinates, m1, m20, mpfMReg, mRegPReg, mpfMRegwoDis, mRegPRegwoDis, msdr_top1, msdr_postMReg, msdr_RegPReg, msdr_postMRegwoDis, msdr_RegPRegwoDis, mtime) in enumerate(zip(template_paths, templates_features, template_fids_WithFuse, meter1, meter20, meterpostMRegfinal, meterRegPostReg, meterpostMRegwoDisfinal, meterRegPostRegwoDis, metersSDR_top1, metersSDR_postMReg, metersSDR_RegPostReg, metersSDR_postMRegwoDis, metersSDR_RegPostRegwoDis, metersTimes)):
            end = time.time()
            afids_ValidPoints_logits = cos(template_features.unsqueeze(1), validMRI_features)
            if args.LocaleSearch:
                afid_SpaceLabel_Restrain_matrix = get_afidCoordSpaceFilter(template_fidcoordinates, validMRI_Pointlists, threshold=args.ValidSelectRadius)  # afidNum * pointsNum
                afids_ValidPoints_logits = afids_ValidPoints_logits * afid_SpaceLabel_Restrain_matrix
            afids_tpok_values, afids_tpok_indexs = torch.topk(afids_ValidPoints_logits, 20, dim=1)
            validMRI_Pointlists = torch.tensor(validMRI_Pointlists)
            afids_topk_points = validMRI_Pointlists[afids_tpok_indexs]  # afidNum , k(topk), 3
            time_match = time.time() - end
            
            end = time.time()

            # whole geometric constraint: 
            #   1.  dist_consistency 
            #   2.  dir_consistency
            #   3.  regisration template
            #   4.  use the registration template to select new valid points
            #   5.  dist_consistency
            #   6.  dir_consistency
            templateMRI_array = np.load(template_path)
            if args.postRegReg:
                afids_PostFinalCoordinates_registration, afids_PostFinalIndex, Reged_TemplateArray, Reged_template_afids = PostProcess_modify_registration(templateMRI_array, validMRI_array, template_fidcoordinates, afids_topk_points)  # afidNum,  3
                validRegSelectedDL, validRegselectPoints = get_eval_validUtils(Reged_template_afids, validMRI_array, selectRadius=args.ValidSelectRadiusReg)
                validMRIRegSelectFeatures, _ = get_ValidMRIPoints_features([validRegSelectedDL], model, device=device) 
                afids_ValidRegSelect_logits = cos(template_features.unsqueeze(1), validMRIRegSelectFeatures[0])   # afidNum * RegSelectPointNum
                if args.LocaleSearch:
                    afid_SpaceLabel_RegSelectRestrain_matrix = get_afidCoordSpaceFilter(Reged_template_afids, validRegselectPoints, threshold=args.ValidSelectRadiusReg)  # afidNum * pointsNum
                    afids_ValidRegSelect_logits = afids_ValidRegSelect_logits * afid_SpaceLabel_RegSelectRestrain_matrix
                afids_topk_RegSelectValues, afids_tpok_RegSelectIndexs = torch.topk(afids_ValidRegSelect_logits, 20, dim=1)
                validRegselectPoints = torch.tensor(validRegselectPoints)
                afids_topk_RegSelectPoints = validRegselectPoints[afids_tpok_RegSelectIndexs]
                afids_RegRegPostReg, afids_RegRegPostRegIndex, _, _ = PostProcess_modify_registration(Reged_TemplateArray, validMRI_array, Reged_template_afids, afids_topk_RegSelectPoints)  # afidNum,  3
                afids_RegPostReg_Dist = torch.norm((torch.tensor(validMRI_golds) - afids_RegRegPostReg).float(), dim=1) # 3
                afids_PostFinalMReg_Dist = torch.norm((torch.tensor(validMRI_golds) - afids_PostFinalCoordinates_registration).float(), dim=1) # 3
            else:
                afids_PostFinalMReg_Dist = torch.zeros((len(torch.tensor(validMRI_golds))), dtype=torch.float32)
                afids_RegPostReg_Dist = torch.zeros((len(torch.tensor(validMRI_golds))), dtype=torch.float32)
            time_post = time.time() - end

            # without dir_consistency
            #  1.  dist_consistency
            #  2.  regisration template
            #  3.  use the registration template to select new valid points
            #  4.  dist_consistency
            if args.woDIRConsistency:
                afids_PostFinalCoordinates_registrationwoDis, afids_PostFinalwoDisIndex, RegedwoDis_TemplateArray, RegedwoDis_template_afids = PostProcess_registration_wo_dirConsistency(templateMRI_array, validMRI_array, template_fidcoordinates, afids_topk_points)  # afidNum,  3
                validRegSelectedwoDisDL, validRegselectwoDisPoints = get_eval_validUtils(RegedwoDis_template_afids, validMRI_array, selectRadius=args.ValidSelectRadiusReg)
                validMRIRegSelectwoDisFeatures, _ = get_ValidMRIPoints_features([validRegSelectedwoDisDL], model, device=device)
                afids_ValidRegSelectwoDis_logits = cos(template_features.unsqueeze(1), validMRIRegSelectwoDisFeatures[0])   # afidNum * RegSelectPointNum
                if args.LocaleSearch:
                    afid_SpaceLabel_RegSelectRestrain_matrix = get_afidCoordSpaceFilter(RegedwoDis_template_afids, validRegselectwoDisPoints, threshold=args.ValidSelectRadiusReg)  # afidNum * pointsNum
                    afids_ValidRegSelectwoDis_logits = afids_ValidRegSelectwoDis_logits * afid_SpaceLabel_RegSelectRestrain_matrix
                afids_topk_RegSelectValues, afids_tpok_RegSelectwoDisIndexs = torch.topk(afids_ValidRegSelectwoDis_logits, 20, dim=1)
                validRegselectwoDisPoints = torch.tensor(validRegselectwoDisPoints)
                afids_topk_RegSelectwoDisPoints = validRegselectwoDisPoints[afids_tpok_RegSelectwoDisIndexs]
                afids_RegRegPostRegwoDis, afids_RegRegPostRegwoDisIndex, _, _ = PostProcess_registration_wo_dirConsistency(RegedwoDis_TemplateArray, validMRI_array, RegedwoDis_template_afids, afids_topk_RegSelectwoDisPoints)  # afidNum,  3

                afids_RegPostRegwoDis_Dist = torch.norm((torch.tensor(validMRI_golds) - afids_RegRegPostRegwoDis).float(), dim=1) # 3
                afids_PostFinalMRegwoDis_Dist = torch.norm((torch.tensor(validMRI_golds) - afids_PostFinalCoordinates_registrationwoDis).float(), dim=1) # 3
            else:
                afids_RegPostRegwoDis_Dist = torch.zeros((len(torch.tensor(validMRI_golds))), dtype=torch.float32)
                afids_PostFinalMRegwoDis_Dist = torch.zeros((len(torch.tensor(validMRI_golds))), dtype=torch.float32)

            
            # compute the mre and sdr
            afids_topk_Dist = torch.norm((torch.tensor(validMRI_golds).unsqueeze(1) - afids_topk_points).float(), dim=2)  # afidNum , k(topk)
            afids_topk_min_Dist, ind_topkmin = torch.min(afids_topk_Dist, dim=1)

            top1_sdr = compute_SDR(afids_topk_Dist[:, 0])
            postMReg_sdr = compute_SDR(afids_PostFinalMReg_Dist)
            RegPostReg_sdr = compute_SDR(afids_RegPostReg_Dist)
            postMRegwoDis_sdr = compute_SDR(afids_PostFinalMRegwoDis_Dist)
            RegPostRegwoDis_sdr = compute_SDR(afids_RegPostRegwoDis_Dist)

            mpfMReg.update(np.array(afids_PostFinalMReg_Dist).round(5)) 
            mRegPReg.update(np.array(afids_RegPostReg_Dist).round(5)) 
            mpfMRegwoDis.update(np.array(afids_PostFinalMRegwoDis_Dist).round(5)) 
            mRegPRegwoDis.update(np.array(afids_RegPostRegwoDis_Dist).round(5)) 

            m1.update(np.array(afids_topk_Dist[:, 0]).round(5))
            m20.update(np.array(afids_topk_min_Dist).round(5))
            msdr_top1.update(top1_sdr)

            msdr_postMReg.update(postMReg_sdr)
            msdr_RegPReg.update(RegPostReg_sdr)
            msdr_postMRegwoDis.update(postMRegwoDis_sdr)
            msdr_RegPRegwoDis.update(RegPostRegwoDis_sdr)
            mtime.update(np.array([time_match, time_post]).round(5))

        progress20.display(i+1)
        progress1.display(i+1)
        progressPostMRegwoDisFinal.display(i+1)
        progressPostMRegFinal.display(i+1)
        progressRegPostRegwoDis.display(i+1)
        progressRegPostReg.display(i+1)

        progressSDR_top1.display(i+1)
        progressSDR_postMRegwoDis.display(i+1)
        progressSDR_postMReg.display(i+1)
        progressSDR_RegPostRegwoDis.display(i+1)
        progressSDR_RegPostReg.display(i+1)
        progressTimes.display(i+1)

    print(f"{meter1[0].name}: \
        \n top1_error: {meter1[0].name}\n({meter1[0].avg})(<mean:{np.mean(meter1[0].avg)}>(<std:{np.mean(meter1[0].std)}>))\n<sdr:{metersSDR_top1[0].avg}>\
        \n postMRegwoDisfinal_error: {meterpostMRegwoDisfinal[0].name}\n({meterpostMRegwoDisfinal[0].avg})(<mean:{np.mean(meterpostMRegwoDisfinal[0].avg)}>(<std:{np.mean(meterpostMRegwoDisfinal[0].std)}>))\n<sdr:{metersSDR_postMRegwoDis[0].avg}>\
        \n postMRegfinal_error: {meterpostMRegfinal[0].name}\n({meterpostMRegfinal[0].avg})(<mean:{np.mean(meterpostMRegfinal[0].avg)}>(<std:{np.mean(meterpostMRegfinal[0].std)}>))\n<sdr:{metersSDR_postMReg[0].avg}>\
        \n RegPostRegwoDisfinal_error: {meterRegPostRegwoDis[0].name}\n({meterRegPostRegwoDis[0].avg})(<mean:{np.mean(meterRegPostRegwoDis[0].avg)}>(<std:{np.mean(meterRegPostRegwoDis[0].std)}>))\n<sdr:{metersSDR_RegPostRegwoDis[0].avg}>\
        \n RegPostRegfinal_error: {meterRegPostReg[0].name}\n({meterRegPostReg[0].avg})(<mean:{np.mean(meterRegPostReg[0].avg)}>(<std:{np.mean(meterRegPostReg[0].std)}>))\n<sdr:{metersSDR_RegPostReg[0].avg}>\n")
    print(f"{meter1[1].name}: \
        \n top1_error: {meter1[1].name}\n({meter1[1].avg})(<mean:{np.mean(meter1[1].avg)}>(<std:{np.mean(meter1[1].std)}>))\n<sdr:{metersSDR_top1[1].avg}>\
        \n postMRegwoDisfinal_error: {meterpostMRegwoDisfinal[1].name}\n({meterpostMRegwoDisfinal[1].avg})(<mean:{np.mean(meterpostMRegwoDisfinal[1].avg)}>(<std:{np.mean(meterpostMRegwoDisfinal[1].std)}>))\n<sdr:{metersSDR_postMRegwoDis[1].avg}>\
        \n postMRegfinal_error: {meterpostMRegfinal[1].name}\n({meterpostMRegfinal[1].avg})(<mean:{np.mean(meterpostMRegfinal[1].avg)}>(<std:{np.mean(meterpostMRegfinal[1].std)}>))\n<sdr:{metersSDR_postMReg[1].avg}>\
        \n RegPostRegwoDisfinal_error: {meterRegPostRegwoDis[1].name}\n({meterRegPostRegwoDis[1].avg})(<mean:{np.mean(meterRegPostRegwoDis[1].avg)}>(<std:{np.mean(meterRegPostRegwoDis[1].std)}>))\n<sdr:{metersSDR_RegPostRegwoDis[1].avg}>\
        \n RegPostRegfinal_error: {meterRegPostReg[1].name}\n({meterRegPostReg[1].avg})(<mean:{np.mean(meterRegPostReg[1].avg)}>(<std:{np.mean(meterRegPostReg[1].std)}>))\n<sdr:{metersSDR_RegPostReg[1].avg}>\n")
    print(f"{meter1[2].name}：\
        \n top1_error: {meter1[2].name}\n({meter1[2].avg})(<mean:{np.mean(meter1[2].avg)}>(<std:{np.mean(meter1[2].std)}>))\n<sdr:{metersSDR_top1[2].avg}>\
        \n postMRegwoDisfinal_error: {meterpostMRegwoDisfinal[2].name}\n({meterpostMRegwoDisfinal[2].avg})(<mean:{np.mean(meterpostMRegwoDisfinal[2].avg)}>(<std:{np.mean(meterpostMRegwoDisfinal[2].std)}>))\n<sdr:{metersSDR_postMRegwoDis[2].avg}>\
        \n postMRegfinal_error: {meterpostMRegfinal[2].name}\n({meterpostMRegfinal[2].avg})(<mean:{np.mean(meterpostMRegfinal[2].avg)}>(<std:{np.mean(meterpostMRegfinal[2].std)}>))\n<sdr:{metersSDR_postMReg[2].avg}>\
        \n RegPostRegwoDisfinal_error: {meterRegPostRegwoDis[2].name}\n({meterRegPostRegwoDis[2].avg})(<mean:{np.mean(meterRegPostRegwoDis[2].avg)}>(<std:{np.mean(meterRegPostRegwoDis[2].std)}>))\n<sdr:{metersSDR_RegPostRegwoDis[2].avg}>\
        \n RegPostRegfinal_error: {meterRegPostReg[2].name}\n({meterRegPostReg[2].avg})(<mean:{np.mean(meterRegPostReg[2].avg)}>(<std:{np.mean(meterRegPostReg[2].std)}>))\n<sdr:{metersSDR_RegPostReg[2].avg}>\n")
    print(f"{meter1[3].name}：\
        \n top1_error: {meter1[3].name}\n({meter1[3].avg})(<mean:{np.mean(meter1[3].avg)}>(<std:{np.mean(meter1[3].std)}>))\n<sdr:{metersSDR_top1[3].avg}>\
        \n postMRegwoDisfinal_error: {meterpostMRegwoDisfinal[3].name}\n({meterpostMRegwoDisfinal[3].avg})(<mean:{np.mean(meterpostMRegwoDisfinal[3].avg)}>(<std:{np.mean(meterpostMRegwoDisfinal[3].std)}>))\n<sdr:{metersSDR_postMRegwoDis[3].avg}>\
        \n postMRegfinal_error: {meterpostMRegfinal[3].name}\n({meterpostMRegfinal[3].avg})(<mean:{np.mean(meterpostMRegfinal[3].avg)}>(<std:{np.mean(meterpostMRegfinal[3].std)}>))\n<sdr:{metersSDR_postMReg[3].avg}>\
        \n RegPostRegwoDisfinal_error: {meterRegPostRegwoDis[3].name}\n({meterRegPostRegwoDis[3].avg})(<mean:{np.mean(meterRegPostRegwoDis[3].avg)}>(<std:{np.mean(meterRegPostRegwoDis[3].std)}>))\n<sdr:{metersSDR_RegPostRegwoDis[3].avg}>\
        \n RegPostRegfinal_error: {meterRegPostReg[3].name}\n({meterRegPostReg[3].avg})(<mean:{np.mean(meterRegPostReg[3].avg)}>(<std:{np.mean(meterRegPostReg[3].std)}>))\n<sdr:{metersSDR_RegPostReg[3].avg}>\n")
    # print(f"All Source and All public：\
    #     \n top1_error: {meter1[-1].name}\n({meter1[-1].avg})(<mean:{np.mean(meter1[-1].avg)}>(<std:{np.mean(meter1[-1].std)}>))\n<sdr:{metersSDR_top1[-1].avg}>\
    #     \n postMRegwoDisfinal_error: {meterpostMRegwoDisfinal[-1].name}\n({meterpostMRegwoDisfinal[-1].avg})(<mean:{np.mean(meterpostMRegwoDisfinal[-1].avg)}>(<std:{np.mean(meterpostMRegwoDisfinal[-1].std)}>))\n<sdr:{metersSDR_postMRegwoDis[-1].avg}>\
    #     \n postMRegfinal_error: {meterpostMRegfinal[-1].name}\n({meterpostMRegfinal[-1].avg})(<mean:{np.mean(meterpostMRegfinal[-1].avg)}>(<std:{np.mean(meterpostMRegfinal[-1].std)}>))\n<sdr:{metersSDR_postMReg[-1].avg}>\
    #     \n RegPostRegwoDisfinal_error: {meterRegPostRegwoDis[-1].name}\n({meterRegPostRegwoDis[-1].avg})(<mean:{np.mean(meterRegPostRegwoDis[-1].avg)}>(<std:{np.mean(meterRegPostRegwoDis[-1].std)}>))\n<sdr:{metersSDR_RegPostRegwoDis[-1].avg}>\
    #     \n RegPostRegfinal_error: {meterRegPostReg[-1].name}\n({meterRegPostReg[-1].avg})(<mean:{np.mean(meterRegPostReg[-1].avg)}>(<std:{np.mean(meterRegPostReg[-1].std)}>))\n<sdr:{metersSDR_RegPostReg[-1].avg}>\n")
    # print(f"All Public: \
    #     \n top1_error: {meter1[0].name}\n({meter1[0].avg})(<mean:{np.mean(meter1[0].avg)}>(<std:{np.mean(meter1[0].std)}>))\n<sdr:{metersSDR_top1[0].avg}>\
    #     \n postMRegwoDisfinal_error: {meterpostMRegwoDisfinal[0].name}\n({meterpostMRegwoDisfinal[0].avg})(<mean:{np.mean(meterpostMRegwoDisfinal[0].avg)}>(<std:{np.mean(meterpostMRegwoDisfinal[0].std)}>))\n<sdr:{metersSDR_postMRegwoDis[0].avg}>\
    #     \n postMRegfinal_error: {meterpostMRegfinal[0].name}\n({meterpostMRegfinal[0].avg})(<mean:{np.mean(meterpostMRegfinal[0].avg)}>(<std:{np.mean(meterpostMRegfinal[0].std)}>))\n<sdr:{metersSDR_postMReg[0].avg}>\
    #     \n RegPostRegwoDisfinal_error: {meterRegPostRegwoDis[0].name}\n({meterRegPostRegwoDis[0].avg})(<mean:{np.mean(meterRegPostRegwoDis[0].avg)}>(<std:{np.mean(meterRegPostRegwoDis[0].std)}>))\n<sdr:{metersSDR_RegPostRegwoDis[0].avg}>\
    #     \n RegPostRegfinal_error: {meterRegPostReg[0].name}\n({meterRegPostReg[0].avg})(<mean:{np.mean(meterRegPostReg[0].avg)}>(<std:{np.mean(meterRegPostReg[0].std)}>))\n<sdr:{metersSDR_RegPostReg[0].avg}>")
    # print(f"One Source: \
    #     \n top1_error: {meter1[1].name}\n({meter1[1].avg})(<mean:{np.mean(meter1[1].avg)}>(<std:{np.mean(meter1[1].std)}>))\n<sdr:{metersSDR_top1[1].avg}>\
    #     \n postMRegwoDisfinal_error: {meterpostMRegwoDisfinal[1].name}\n({meterpostMRegwoDisfinal[1].avg})(<mean:{np.mean(meterpostMRegwoDisfinal[1].avg)}>(<std:{np.mean(meterpostMRegwoDisfinal[1].std)}>))\n<sdr:{metersSDR_postMRegwoDis[1].avg}>\
    #     \n postMRegfinal_error: {meterpostMRegfinal[1].name}\n({meterpostMRegfinal[1].avg})(<mean:{np.mean(meterpostMRegfinal[1].avg)}>(<std:{np.mean(meterpostMRegfinal[1].std)}>))\n<sdr:{metersSDR_postMReg[1].avg}>\
    #     \n RegPostRegwoDisfinal_error: {meterRegPostRegwoDis[1].name}\n({meterRegPostRegwoDis[1].avg})(<mean:{np.mean(meterRegPostRegwoDis[1].avg)}>(<std:{np.mean(meterRegPostRegwoDis[1].std)}>))\n<sdr:{metersSDR_RegPostRegwoDis[1].avg}>\
    #     \n RegPostRegfinal_error: {meterRegPostReg[1].name}\n({meterRegPostReg[1].avg})(<mean:{np.mean(meterRegPostReg[1].avg)}>(<std:{np.mean(meterRegPostReg[1].std)}>))\n<sdr:{metersSDR_RegPostReg[1].avg}>")
    # print(f"One Source and All public：\
    #     \n top1_error: {meter1[2].name}\n({meter1[2].avg})(<mean:{np.mean(meter1[2].avg)}>(<std:{np.mean(meter1[2].std)}>))\n<sdr:{metersSDR_top1[2].avg}>\
    #     \n postMRegwoDisfinal_error: {meterpostMRegwoDisfinal[2].name}\n({meterpostMRegwoDisfinal[2].avg})(<mean:{np.mean(meterpostMRegwoDisfinal[2].avg)}>(<std:{np.mean(meterpostMRegwoDisfinal[2].std)}>))\n<sdr:{metersSDR_postMRegwoDis[2].avg}>\
    #     \n postMRegfinal_error: {meterpostMRegfinal[2].name}\n({meterpostMRegfinal[2].avg})(<mean:{np.mean(meterpostMRegfinal[2].avg)}>(<std:{np.mean(meterpostMRegfinal[2].std)}>))\n<sdr:{metersSDR_postMReg[2].avg}>\
    #     \n RegPostRegwoDisfinal_error: {meterRegPostRegwoDis[2].name}\n({meterRegPostRegwoDis[2].avg})(<mean:{np.mean(meterRegPostRegwoDis[2].avg)}>(<std:{np.mean(meterRegPostRegwoDis[2].std)}>))\n<sdr:{metersSDR_RegPostRegwoDis[2].avg}>\
    #     \n RegPostRegfinal_error: {meterRegPostReg[2].name}\n({meterRegPostReg[2].avg})(<mean:{np.mean(meterRegPostReg[2].avg)}>(<std:{np.mean(meterRegPostReg[2].std)}>))\n<sdr:{metersSDR_RegPostReg[2].avg}>")
    # print(f"Ten Source and All public：\
    #     \n top1_error: {meter1[3].name}\n({meter1[3].avg})(<mean:{np.mean(meter1[3].avg)}>(<std:{np.mean(meter1[3].std)}>))\n<sdr:{metersSDR_top1[3].avg}>\
    #     \n postMRegwoDisfinal_error: {meterpostMRegwoDisfinal[3].name}\n({meterpostMRegwoDisfinal[3].avg})(<mean:{np.mean(meterpostMRegwoDisfinal[3].avg)}>(<std:{np.mean(meterpostMRegwoDisfinal[3].std)}>))\n<sdr:{metersSDR_postMRegwoDis[3].avg}>\
    #     \n postMRegfinal_error: {meterpostMRegfinal[3].name}\n({meterpostMRegfinal[3].avg})(<mean:{np.mean(meterpostMRegfinal[3].avg)}>(<std:{np.mean(meterpostMRegfinal[3].std)}>))\n<sdr:{metersSDR_postMReg[3].avg}>\
    #     \n RegPostRegwoDisfinal_error: {meterRegPostRegwoDis[3].name}\n({meterRegPostRegwoDis[3].avg})(<mean:{np.mean(meterRegPostRegwoDis[3].avg)}>(<std:{np.mean(meterRegPostRegwoDis[3].std)}>))\n<sdr:{metersSDR_RegPostRegwoDis[3].avg}>\
    #     \n RegPostRegfinal_error: {meterRegPostReg[3].name}\n({meterRegPostReg[3].avg})(<mean:{np.mean(meterRegPostReg[3].avg)}>(<std:{np.mean(meterRegPostReg[3].std)}>))\n<sdr:{metersSDR_RegPostReg[3].avg}>")
    # print(f"All Source and All public：\
    #     \n top1_error: {meter1[-1].name}\n({meter1[-1].avg})(<mean:{np.mean(meter1[-1].avg)}>(<std:{np.mean(meter1[-1].std)}>))\n<sdr:{metersSDR_top1[-1].avg}>\
    #     \n postMRegwoDisfinal_error: {meterpostMRegwoDisfinal[-1].name}\n({meterpostMRegwoDisfinal[-1].avg})(<mean:{np.mean(meterpostMRegwoDisfinal[-1].avg)}>(<std:{np.mean(meterpostMRegwoDisfinal[-1].std)}>))\n<sdr:{metersSDR_postMRegwoDis[-1].avg}>\
    #     \n postMRegfinal_error: {meterpostMRegfinal[-1].name}\n({meterpostMRegfinal[-1].avg})(<mean:{np.mean(meterpostMRegfinal[-1].avg)}>(<std:{np.mean(meterpostMRegfinal[-1].std)}>))\n<sdr:{metersSDR_postMReg[-1].avg}>\
    #     \n RegPostRegwoDisfinal_error: {meterRegPostRegwoDis[-1].name}\n({meterRegPostRegwoDis[-1].avg})(<mean:{np.mean(meterRegPostRegwoDis[-1].avg)}>(<std:{np.mean(meterRegPostRegwoDis[-1].std)}>))\n<sdr:{metersSDR_RegPostRegwoDis[-1].avg}>\
    #     \n RegPostRegfinal_error: {meterRegPostReg[-1].name}\n({meterRegPostReg[-1].avg})(<mean:{np.mean(meterRegPostReg[-1].avg)}>(<std:{np.mean(meterRegPostReg[-1].std)}>))\n<sdr:{metersSDR_RegPostReg[-1].avg}>")
def compute_SDR(dis):
     """
    Compute the successful detection rates (SDR) for distances from 1 to 10 mm.

    Args:
        dis: Array of distances.

    Returns:
        Array of SDR values rounded to 4 decimal places.
    """
    sdr_result = []
    # 循环计算：
    for i in [2,4,8,10]:
        i_sdr = len(dis[dis<i]) / len(dis)
        sdr_result.append(i_sdr)
    return np.array(sdr_result).round(4)

def get_eval_validUtils(templateAfids, validArray, selectRadius=25):
    """
    1. Determine the candidate point range using the transformed template fiducial points and select dense candidate points.
    2. Build a Dataset for the reselected validation data and its candidate points, and extract embeddings.

    Args:
        templateAfids: Fiducial points of the template.
        validArray: Validation data array.
        selectRadius: Radius for selecting candidate points.

    Returns:
        A tuple containing the validation data loader and the selected candidate points.
    """
    valid_selectPoints = validPoints_denseSample(templateAfids, validSelectRadius=selectRadius)
    validSelectedDL, _ = getInference_dataloaders([validArray], valid_selectPoints, fileType='array')
    return validSelectedDL[0], valid_selectPoints[0]


def validPoints_denseSample(templateafids, validSampleNum=1, margin=1, validSelectRadius=25):
    """
    Estimate the approximate area of the anatomical fiducial points to be located based on the positions of the template anatomical fiducial points, and perform dense sampling within the area.
    1. Sample at equal intervals of 2 pixel units within 25 pixel units around the template anatomical fiducial points.

    Args:
        templateafids: Template anatomical fiducial points.
        validSampleNum: Number of validation samples.
        margin: Sampling interval.
        validSelectRadius: Radius for selecting candidate points.

    Returns:
        List of validation point lists.
    """
    validDensePoints = []
    templateafids = np.array(templateafids).astype(int)
    # 对所有的模板解剖标志进行循环
    for templateafid in templateafids:
       # Generate equally spaced sample points for each coordinate
        # First, calculate the upper and lower limits of each dimension
        min = templateafid-validSelectRadius
        min[min<0] = 0
        max = templateafid+validSelectRadius
        max[max>255]=255
        points = [[i,j,k] for i in range(min[0], max[0], margin) for j in range(min[1], max[1], margin) for k in range(min[2], max[2], margin)]
        validDensePoints += points
          
    # Remove duplicate points from the array
    validDensePoints = np.unique(validDensePoints, axis=0).tolist()
   
    # Add corresponding data for each validation data as candidate anatomical fiducial points during validation
    valid_pointLists = [validDensePoints for _ in range(validSampleNum)]
    return valid_pointLists


def get_afidCoordSpaceFilter(template_fidcoordinates, validMRIs_Pointlists, threshold=30):
    """
    Compute the distance between each fiducial point and each point. If the distance is less than the threshold, the value is 1, otherwise 0.

    Args:
        template_fidcoordinates: Fiducial coordinates of the template.
        validMRIs_Pointlists: Point lists of the valid MRIs.
        threshold: Distance threshold.

    Returns:
        A matrix of shape (afidNum * pointsNum).
    """
    pointsNum = len(validMRIs_Pointlists)
    afidNum = len(template_fidcoordinates)
    template_Coordinates = torch.tensor(template_fidcoordinates)
    validMRI_PointCoordinates = torch.tensor(validMRIs_Pointlists).T
   
    template_Coordinates = template_Coordinates.reshape(-1) # (afidsNum*3)
    boradcast_template_Coordinates = template_Coordinates.repeat(pointsNum, 1).T  # （3*afidsNum）* pointsNum
    
    boradcast_queue_Coordinates = validMRI_PointCoordinates.repeat(afidNum, 1) # （3*afidsNum）* pointsNum

    # Compare the broadcast queue with the broadcast maximum and minimum ranges to obtain a matrix of True or False values
    MinusCoordinates = boradcast_queue_Coordinates - boradcast_template_Coordinates
    afid_SpaceLabelRestrain_indexMatrix = torch.norm(MinusCoordinates.reshape(-1, 3, pointsNum), dim=1) <= threshold# afidsNum * pointsNum


    # Initialize the restraint matrix
    restrain_matrix = torch.ones((afidNum, pointsNum)) * 0.01  # 防止在验证时没有被抑制的点不足20这样需要其他被抑制的点来补齐，但是因为被抑制的点都一样，会从头开始选择，这样会产生较大的偏差
    restrain_matrix[afid_SpaceLabelRestrain_indexMatrix] = 1
    return restrain_matrix


def get_AverageMeter_progressMeter(args, template_paths):
    if args.dataset == 'sanbo':
        template_paths.append('fuseTemplates')
    else:
        template_paths = ['fuseTemplates', 'source1template', 'fuseTempAndSource1', 'fuseTempAndSource10', 'fuseTempAndSource20']

    # template_paths =  fuse_names 
    mre_dis1_Average_list = [AverageMeter_Eval(f"{get_nameIn(template_path, 0)}", ":6.2f") for template_path in template_paths]
    mre_dis20_Average_list = [AverageMeter_Eval(f"{get_nameIn(template_path, 0)}", ":6.2f") for template_path in template_paths]
    mre_finePostMReg_Average_list = [AverageMeter_Eval(f"{get_nameIn(template_path, 0)}", ":6.2f") for template_path in template_paths]
    mre_RegPostReg_Average_list = [AverageMeter_Eval(f"{get_nameIn(template_path, 0)}", ":6.2f") for template_path in template_paths]
    mre_finePostMRegwoDis_Average_list = [AverageMeter_Eval(f"{get_nameIn(template_path, 0)}", ":6.2f") for template_path in template_paths]
    mre_RegPostRegwoDis_Average_list = [AverageMeter_Eval(f"{get_nameIn(template_path, 0)}", ":6.2f") for template_path in template_paths]
    sdr_top1_Average_list = [AverageMeter_Eval(f"{get_nameIn(template_path, 0)}", ":6.2f") for template_path in template_paths]
    sdr_postMReg_Average_list = [AverageMeter_Eval(f"{get_nameIn(template_path, 0)}", ":6.2f") for template_path in template_paths]
    sdr_RegPostReg_Average_list = [AverageMeter_Eval(f"{get_nameIn(template_path, 0)}", ":6.2f") for template_path in template_paths]
    sdr_postMRegwoDis_Average_list = [AverageMeter_Eval(f"{get_nameIn(template_path, 0)}", ":6.2f") for template_path in template_paths]
    sdr_RegPostRegwoDis_Average_list = [AverageMeter_Eval(f"{get_nameIn(template_path, 0)}", ":6.2f") for template_path in template_paths]
    time_Average_list = [AverageMeter_Eval(f"{get_nameIn(template_path, 0)}", ":6.2f") for template_path in template_paths]

    return (template_paths, mre_dis1_Average_list, mre_dis20_Average_list, mre_finePostMReg_Average_list, mre_RegPostReg_Average_list,mre_finePostMRegwoDis_Average_list, mre_RegPostRegwoDis_Average_list,\
             sdr_top1_Average_list, sdr_postMReg_Average_list, sdr_RegPostReg_Average_list, sdr_postMRegwoDis_Average_list, sdr_RegPostRegwoDis_Average_list, time_Average_list)




def get_ValidMRIPoints_features(valid_dataloaders, model, device='cpu'):
    """
    Use the model to encode the data in each validation dataloader in batches, and save the results. Finally, obtain a list of features.

    Args:
        valid_dataloaders: Validation dataloaders.
        model: The model for feature extraction.
        device: Device to run the model on.

    Returns:
        A tuple containing the list of validation MRI point features and the mean time for feature extraction.
    """
    validMRIPointsFeature_list = []
    end = time.time()
    for valid_dataloader in valid_dataloaders:
        intermediate_features = torch.tensor([])
        for crops in valid_dataloader:
            torch.cuda.empty_cache()
            intermediate_feature = model(crops.to(device)).cpu().detach()  # batchsize*512
            intermediate_features = torch.cat([intermediate_features, intermediate_feature], dim=0)  # (n*batchsize) * 512
            
        validMRIPointsFeature_list.append(intermediate_features)
    meanTimesfeatureExtract = (time.time() - end)/len(valid_dataloaders)
    return validMRIPointsFeature_list, meanTimesfeatureExtract

def get_template_Cropfeature_Coordinates_sanbo(template_dataloaders, template_fids, fuseTemplate_dataloaders, model, fidNum=32, fuse_template_list=[1,2,6,7], template_ind=2, device='cpu'):
    """
    1. Extract the features around 35 feature points of n templates using the model and save them in a list.
    2. Perform feature fusion on the templates in the feature space (average the features of these templates at the reference points as a template for testing based on the templates that performed well in the past).
        a. A template containing the original data (templates 1, 2, 6, 7, 9) is required.
        b. A template without the original data is required for template matching without relying on the data.

    Args:
        template_dataloaders: Dataloaders for the templates.
        template_fids: Fiducial coordinates of the templates.
        fuseTemplate_dataloaders: Dataloaders for the fusion templates.
        model: The model for feature extraction.
        fidNum: Number of fiducial points.
        fuse_template_list: List of templates for fusion.
        template_ind: Index of the template.
        device: Device to run the model on.

    Returns:
        A tuple containing the features of all template feature points and the fiducial coordinates of the fusion templates.
    """
    MRI_features = []
    
    
    featureSum_withOut_sourceTemplate = 0
    print("Public template feature extraction...")
    for i, dl_tempalte in enumerate(template_dataloaders):
        crops = next(iter(dl_tempalte)).to(device)
        features = model(crops).cpu().detach()  # afidsNum * encoderDim
        if i in fuse_template_list:
            featureSum_withOut_sourceTemplate += features
            MRI_features.append(features)
        # MRI_features.append(features)
    fuseFeaturesWithOutSource = featureSum_withOut_sourceTemplate/(len(fuse_template_list))
    MRI_features.append(fuseFeaturesWithOutSource)
    
    template_fids_np = np.array(template_fids)
    template_32fids = template_fids_np[:, :fidNum, :].tolist()  
    fuse_template_32fids = []
    fuse_template_32fids.append(template_32fids[0])
    fuse_template_32fids.append(template_32fids[1])
    fuse_template_32fids.append(template_32fids[2])
    fuse_template_32fids.append(template_32fids[template_ind])


    return MRI_features, fuse_template_32fids
def get_template_Cropfeature_Coordinates(template_dataloaders, template_fids, fuseTemplate_dataloaders, model, fidNum=32, fuse_template_list=[1,2,6,7], template_ind=2, device='cpu'):
    """
    1. Extract the features around 35 feature points of n templates using the model and save them in a list.
    2. Perform feature fusion on the templates in the feature space (average the features of these templates at the reference points as a template for testing based on the templates that performed well in the past).
        a. A template containing the original data (templates 1, 2, 6, 7, 9) is required.
        b. A template without the original data is required for template matching without relying on the data.

    Args:
        template_dataloaders: Dataloaders for the templates.
        template_fids: Fiducial coordinates of the templates.
        fuseTemplate_dataloaders: Dataloaders for the fusion templates.
        model: The model for feature extraction.
        fidNum: Number of fiducial points.
        fuse_template_list: List of templates for fusion.
        template_ind: Index of the template.
        device: Device to run the model on.

    Returns:
        A tuple containing the features of all template feature points and the fiducial coordinates of the fusion templates.
    """
    MRI_features = []
    
    
    featureSum_withOut_sourceTemplate = 0
    print("Public template feature extraction...")
    for i, dl_tempalte in enumerate(template_dataloaders):
        crops = next(iter(dl_tempalte)).to(device)
        features = model(crops).cpu().detach()  # afidsNum * encoderDim
        if i in fuse_template_list:
            featureSum_withOut_sourceTemplate += features
        # MRI_features.append(features)
    fuseFeaturesWithOutSource = featureSum_withOut_sourceTemplate/(len(fuse_template_list))
    MRI_features.append(fuseFeaturesWithOutSource)

    featureSum_fuseTemplate = 0
    print("Source template feature extraction...")
    for i, dl_fuseNpys in enumerate(fuseTemplate_dataloaders):
        crops = next(iter(dl_fuseNpys)).to(device)
        features = model(crops).cpu().detach()
        featureSum_fuseTemplate += features
        if i == 0:
            feature_fuseTemplate1 = features
            # MRI_features.append(feature_fuseTemplate1)
        if i == 9:
            feature_fuseTemplate10 = featureSum_fuseTemplate/10
            # MRI_features.append(feature_fuseTemplate10)
    feature_fuseTemplate20 = featureSum_fuseTemplate/(len(fuseTemplate_dataloaders))
    # MRI_features.append(feature_fuseTemplate20)

    # Combine the features of the template and the fused data template with a 0.5:0.5 ratio
    template_and_fuse_feature1 = 0.5 * fuseFeaturesWithOutSource + 0.5 * feature_fuseTemplate1
    MRI_features.append(feature_fuseTemplate1)
    MRI_features.append(template_and_fuse_feature1)
    template_and_fuse_feature10 = 0.5 * fuseFeaturesWithOutSource + 0.5 * feature_fuseTemplate10
    MRI_features.append(template_and_fuse_feature10)
    template_and_fuse_feature20 = 0.5 * fuseFeaturesWithOutSource + 0.5 * feature_fuseTemplate20
    MRI_features.append(template_and_fuse_feature20)

    # Set the gold standard points for the fused templates for local calculation during validation to reduce the amount of computation
    template_fids_np = np.array(template_fids)
    template_32fids = template_fids_np[:, :fidNum, :].tolist()  

    fuse_template_32fids = []
    fuse_template_32fids.append(template_32fids[template_ind])
    fuse_template_32fids.append(template_32fids[template_ind])
    fuse_template_32fids.append(template_32fids[template_ind])
    fuse_template_32fids.append(template_32fids[template_ind])
    fuse_template_32fids.append(template_32fids[template_ind])


    print(f"few shot versions and corresponding afid list: {len(MRI_features)}, {len(fuse_template_32fids)}")
    return MRI_features, fuse_template_32fids
