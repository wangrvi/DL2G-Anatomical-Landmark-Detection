'''
DL2G 

feature point crop encoder train
'''

import os
import sys
import time
import torch
import random
import warnings
import argparse
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(__file__))
import utils.util as util
import utils.logger as logger
import model.Resnet as Resnet
from src.eval import eval_one_fold
import model.MoCo_builder as MoCo_builder
from utils.misc import AverageMeter, ProgressMeter, save_checkpoint
from datasets.dataload import get_train_datasets, get_eval_datasets_onefold

def main():
    # args
    args = parase()
    dataset = args.dataset
    config = util.load_config_from_yaml(args.config)
    config = config[dataset]
    args.constMean = config['norm'][0]
    mode = args.mode
    if dataset == 'sanbo':
        args.preScreenSize = config['preScreenSize']
        args.fidNum = config['fidNum']
        args.augnum = config['augnum']
    # path
    augnum = len(args.AugWeights)
    expPropetry = 'contrast/'
    theme = f'{expPropetry}/{args.mode}/aug{augnum}_weights{args.AugWeights}/trainR_{args.TrainRestrainRadius}_ValidR_{args.LocaleSearch}_{args.ValidSelectRadius}/'
    postfix_log = f"{args.arch}_AugRotation:{args.AugRotation}_AugMask:{args.AugMask}_AugScale:{args.AugScale}_screenAug:{args.screenAug}_screenAugNum:{args.augnum}_"
    postfix_ckpt = f"AugRotation:{args.AugRotation}_AugMask:{args.AugMask}_AugScale:{args.AugScale}/{args.arch}_{args.moco_dim}_lr{args.lr}_mlp_{args.mlp}"
    postfix_ = os.path.join(theme, postfix_log)

    output_dir = os.path.join(args.output_dir, dataset)

    log_dir = os.path.join(output_dir, "Logs/")
    args.checkpoint_dir = os.path.join(output_dir, "checkpoints", theme, postfix_ckpt)
    args.plot_dir = os.path.join(output_dir, "figure", postfix_)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True) 
    os.makedirs(args.plot_dir, exist_ok=True)
    # log
    logger.set_log(postfix_, log_dir)

    print("outputPath: ", theme, "\n")
    print("checkpoint path: ", args.checkpoint_dir, '\n')
    print("log path: ", os.path.join(log_dir, postfix_), '\n')
    print("plot path: ", args.plot_dir, '\n')

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    # seed
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )
    if args.gpu is not None:
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')

    if args.resume:
        print("resume from checkpoint", args.resume, "\n")
    if args.LocaleSearch:
        print("ValidSelectRadius", args.ValidSelectRadius, "\n")

    print(f"loacte {args.fidNum} landmark points", "\n")
    print(f"use postregreg: {args.postRegReg}", "\n")
    print(f"use dir consistency: {args.woDIRConsistency}", "\n")

    # dataset

    # model
    print(f"=> creating model '{args.arch}'")
    if args.arch == 'resnet18':
        resnet = Resnet.resnet18
    elif args.arch == 'resnet10':
        resnet = Resnet.resnet10
    
    model = MoCo_builder.MoCo(resnet, 
                              dim=args.moco_dim, #"feature dimension (default: 128)"
                              m=args.moco_m, #"moco momentum of updating key encoder (default: 0.999)"
                              K=args.moco_k, #"queue size; number of negative keys (default: 65536)"
                              T=args.moco_t,  #softmax temperature (default: 0.07)
                              mlp=args.mlp, #MoCo v2
                              TrainRestrainRadius=args.TrainRestrainRadius,
                              gpu=args.gpu,
                              )
    model = model.to(device)
    cudnn.benchmark = True
    # train
    if mode != 'eval':
        # print("train")
        train_ds = get_train_datasets(args, config)
        eval_utils = get_eval_datasets_onefold(args, config)

        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0, last_epoch=-1)
        for epoch in range(args.start_epoch, args.epochs):
            epoch_avglosses = train(train_dl, model, optimizer, epoch, args)
            
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    'losses': epoch_avglosses,
                },
                is_best=False,
                filename=args.checkpoint_dir + f"_checkpoint_{epoch:04d}.pth.tar",
            )

            eval_one_fold(model.get_submodule('encoder_q'), eval_utils, args, config)
            scheduler.step()
    else:
        print("FoldValidation")

def train(train_dl, model, optimizer, epoch, args):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    if args.gpu is not None:
        memory = AverageMeter("max mem", ":.0f")
    progress = ProgressMeter(
        len(train_dl),
        [batch_time, data_time, losses, memory],
        prefix="Epoch: [{}]".format(epoch),
    )
    if args.gpu is not None:
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')

    model.train()
    end = time.time()
    for i, crops in enumerate(train_dl):
        # if i < 2:
        #     plt_plot_batchsize(i+1, crops['sourceCrop'], crops['augCrops'], plot_batchsize=crops['augCrops'].shape[0], savepath=f'{logCheckSavePath}/{foldername}/Logs/picture/{args.arch}_{args.moco_dim}_/{i}batchsize.jpg')
        data_time.update(time.time() - end)
        if i < 1:
            print("check data shape is right or not ", crops['sourceCrop'].shape, crops['augCrops'].shape)
        target = torch.cat([crops['augLabels'], torch.zeros((crops['sourceCrop'].shape[0], args.moco_k), dtype=torch.float32)], dim=1)
        
        crops['sourceCrop'] = crops['sourceCrop'].to(device)
        crops['augCrops'] = crops['augCrops'].to(device)
        crops['anchorCoords'] = crops['anchorCoords'].to(device)
        target = target.to(device)
        
        output, target = model(crop_q=crops['sourceCrop'], crop_aug=crops['augCrops'], crop_Coordinates=crops['anchorCoords'], target=target)
        # loss = criterion(output, target)  正确的损失函数应该是使用log_softmax然后乘以target
        loss = -(F.log_softmax(output, dim=1) * target).sum(dim=1).mean()

        losses.update(loss.item(), crops['sourceCrop'].size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.gpu is not None:
            memory.update(torch.cuda.max_memory_allocated(device=device) / 1024.0 / 1024.0)
        if i % args.print_freq == 0:
            progress.display(i)
    return losses.avg

def parase():
    parser = argparse.ArgumentParser(description="DL2G")
    # 使用占位符替换具体路径
    parser.add_argument("--config", type=str, default="<CONFIG_FILE_PATH>", help="config file path")
    parser.add_argument("--dataset", type=str, default="sanbo", help="dataset name")
    parser.add_argument('--output_dir', default="<OUTPUT_DIR_PATH>", help='path where to save, empty for no saving')

    parser.add_argument("--mode", type=str, default="train", help="train or eval")
    # parser.add_argument("--mode", type=str, default="test", help="train or eval")
    # parser.add_argument("--mode", type=str, default="eval", help="train or eval")
    # parser.add_argument('--FoldValidation', action="store_true", default=False, help="mode: 3 fold cross Validation or Train ")
    parser.add_argument("--gpu", default=3, type=int, help="GPU id to use.")

    # model
    parser.add_argument("-a", "--arch", metavar="ARCH", default="resnet10", help="base_model")
    parser.add_argument("--moco-dim", default=512, type=int, help="feature dimension (default: 128)")
    parser.add_argument("--moco-k", default=65536, type=int, help="queue size; number of negative keys (default: 65536)")
    parser.add_argument("--moco-m", default=0.999, type=float, help="moco momentum of updating key encoder (default: 0.999)")
    parser.add_argument("--moco-t", default=0.07, type=float, help="softmax temperature (default: 0.07)")
    # options for moco v2
    parser.add_argument("--mlp", action="store_true", default=True, help="use mlp head")

    # data
    parser.add_argument("--fidNum", default=32, type=int, help="number of fiducial points")
    parser.add_argument("--cropSize", default=32, type=int, help="crop size")

    # training
    parser.add_argument("--epochs", default=2, type=int, metavar="N", help="number of total epochs to run, multi epoch is effect")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="manual epoch number (useful on restarts)")
    parser.add_argument("-b", "--batch-size", default=256, type=int, metavar="N", help="mini-batch size (default: 256), this is the total batch size of all GPUs")
    parser.add_argument("--lr", "--learning-rate", default=0.001, type=float, metavar="LR", help="initial learning rate", dest="lr")
    parser.add_argument("--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 4)")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument("--weight-decay", default=1e-4, type=float, metavar="W", help="weight decay (default: 1e-4)")
    parser.add_argument("--seed", default=1314520, type=int, help="seed for initializing training. ")
    parser.add_argument("--print-freq", default=10, type=int, metavar="N", help="print frequency (default: 10)")

    # eval
    parser.add_argument("--resume", default="", type=str, metavar="PATH", help="path to latest checkpoint (default: none)")
    parser.add_argument("--batchSize-eval", default=3000, type=int, help="batch size for eval")
    # dist
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--rank", default=-1, type=int, help="node rank for distributed training")
    parser.add_argument("--dist-url", default="tcp://12345", type=str, help="url used to set up distributed training")
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")

    # method setting
    parser.add_argument("--TrainRestrainRadius", default=30, type=int, help="Train Restrain Radius")
    parser.add_argument("--LocaleSearch", action="store_true", default=True, help="Locale Search")
    parser.add_argument("--ValidSelectRadius", default=30, type=int, help="Valid Select Radius")
    parser.add_argument("--ValidSelectRadiusReg", default=15, type=int, help="Valid Select Radius Reg")

    parser.add_argument("--AugRotation", action="store_true", default=True, help="Aug Rotation")
    parser.add_argument("--AugMask", action="store_true", default=True, help="Aug Mask")
    parser.add_argument("--AugScale", action="store_true", default=True, help="Aug Scale")
    parser.add_argument("--AugWeights", default=[0.33, 0.34, 0.33], type=list, help="Aug Weights")

    parser.add_argument("--RotationParames", default=40, type=int, help="Rotation Params")

    parser.add_argument("--woDIRConsistency", action="store_true", default=False, help="wo DIR Consistency")
    parser.add_argument("--postRegReg", action="store_true", default=True, help="post Reg Reg")

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
    args = parser.parse_args()
    return args

if __name__ =="__main__":
    main()