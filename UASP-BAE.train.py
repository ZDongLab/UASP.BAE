# -*- coding: utf-8 -*-
"""
Created by @author: Carlson Zhang. Xi'an Jiaotong University.
"""
import os
import argparse
import torch
from torch.cuda.amp import GradScaler, autocast
from torch import nn
#from torch.nn.parallel import DistributedDataParallel as DDP
#from torch.utils.data.distributed import DistributedSampler
from torchsummary import summary
import torch.optim
from torch.optim import lr_scheduler
from srcfunc.metrics.EvaluationMAE import EvaluationMAE
from srcfunc.models import AgeJoiner, create_model, prepare_device
from srcfunc.loss_functions.losses import  ACLossGs
from srcfunc.helper_functions.helper_functions import ModelEma, savetenmodel, add_weight_decay
from srcfunc.helper_functions.logging import setup_logging
from srcfunc.datasets.toothopgdata import toothopg, GussiCut, age_names_label_dict
#from tqdm import tqdm
from pathlib import Path
import datetime
run_id = datetime.datetime.now().strftime(r'%m%d_%H.%M.%S') # datetime.now().strftime(r'%y%m%d_%H%M%S')
import pandas as pd

__all__=["tresnet_m", "tresnet_l", "tresnet_xl",
        "ttresnet_m", "ttresnet_l", "ttresnet_xl",
         "res2net50", "res2next50", 
         "cmt_ti","cmt_xs","cmt_s","cmt_b",
         "coatnet_0", "coatnet_1","coatnet_2","coatnet_3","coatnet_4",
         "m_crossvit", "m_t2tvit", "m_vit",
         "SwinTransformer_m","SwinTransformer_L","SwinTransformer_XL",
         "UASP_BAE_net0","UASP_BAE_net1","UASP_BAE_net3","UASP_BAE_net4"
        ]

parser = argparse.ArgumentParser(description='PyTorch UASP-BAE Training')
parser.add_argument('--data_name', help='path to dataset', type=str,
                    default ="5-24" 
                    )
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--model_name', default='UASP_BAE_net1')
parser.add_argument('--model_path', default=None, type=str)
parser.add_argument('--num_classes', default=1) 
parser.add_argument('--in_chans', default=1, type=int,
                    help='input image chanel (default: 1)')
parser.add_argument('--image_size', default=(448, 448), type=tuple,
                    help='input image size (default: 448)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    help='mini-batch size (default: 16)')
parser.add_argument('-j', '--workers', default=32, type=int,
                    help='number of data loading workers (default: 16)')
parser.add_argument('--thre', default=0.8, type=float,
                    help='threshold value')
parser.add_argument('--print-freq', '-p', default=64, type=int,
                    help='print frequency (default: 64)')

parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
parser.add_argument('--n_gpu_use', default=2, type=int,
                    help='number of gpus per node')
parser.add_argument('--local_rank', default=-1, type=int,
                    help='ranking within the nodes')

args = parser.parse_args()  

def main():
    args.do_bottleneck_head = False
    device, device_ids = prepare_device(args.n_gpu_use)
    agerange = args.data_name.strip("ab").split("-")
    agerange = (int(agerange[0])-1,int(agerange[1])+1)

    logger = setup_logging("./outputs/{}/{}".format(args.data_name, args.model_name),namemask="{}-age".format(args.model_name))
    savemodel_dir = Path("./models/{}/{}/".format(args.data_name, args.model_name))
    savemodel_dir.mkdir(parents=True, exist_ok=True)

    # Setup model
    print('{} creating model...'.format(args.model_name))
    backbone = create_model(args).cuda()
    if args.model_path is not None:  # make sure to load pretrained model
        checkpoint = torch.load(args.model_path, map_location=torch.device("cpu"))
        filtered_dict = {k: v for k, v in checkpoint["state_dict"]['model'].items() if
                         (k in backbone.state_dict() and 'head.fc' not in k)}
        backbone.load_state_dict(filtered_dict, strict=False)
    
    # get backbone without fc
    backbone.forward = backbone.forward_features

    #del backbone.avgpool
    num_features = backbone.num_features
    del backbone.head
    #del backbone.fc
    model = AgeJoiner(backbone, n_channels=num_features, agegsch=int((agerange[1]-agerange[0])*2)+1 ).cuda()
    print('done\n') 
    logger.info(model)
    # Pytorch Data loader
    imgs_dir, label_csv = "./datasets/train","./datasets/train.%s.csv"%(args.data_name)
    #ToothCls = toothopg(imgs_dir, label_csv)  
    input_imgsize = args.image_size
    logger.info("imagesize = (%d,%d)"%input_imgsize)
    #ToothCls_train = toothopg(imgs_dir, label_csv, imagesize=(args.image_size, args.image_size), tempmemary=False)
    ToothCls_train = toothopg(imgs_dir, label_csv, imagesize=input_imgsize, tempmemary=False, agerange=agerange,mapnum=int((agerange[1]-agerange[0])*2)+1)
    train_loader = torch.utils.data.DataLoader(
        ToothCls_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    imgs_dir, label_csv = "./datasets/val","./datasets/val.%s.csv"%(args.data_name)
    #ToothCls_val = toothopg(imgs_dir, label_csv, imagesize=(args.image_size, args.image_size), tempmemary=False)
    ToothCls_val = toothopg(imgs_dir, label_csv, imagesize=input_imgsize, tempmemary=False, agerange=agerange,mapnum=int((agerange[1]-agerange[0])*2)+1)
    val_loader = torch.utils.data.DataLoader(
        ToothCls_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
        
    # Actuall Training
    ema = ModelEma(model, 0.9997)  # 0.9997^641=0.82
    # set optimizer
    Epochs = 100
    Stop_epoch = 100
    weight_decay = 1e-4
    parameters = add_weight_decay(model, weight_decay)
    criterion_age = nn.HuberLoss()
    criterion_agegs = ACLossGs(gamma_neg=4, gamma_pos=0,clip=0.05)
    optimizer = torch.optim.Adam(params=parameters, lr=args.lr, weight_decay=0)  # true wd, filter_bias_and_bn
    if args.model_path is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    
    ## DDP setting

    ### others params
    steps_per_epoch = len(train_loader)
    scaler = GradScaler()
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=steps_per_epoch, epochs=Epochs,
                                        pct_start=0.2)
    savemodelinfo = savetenmodel(savemodel_dir, save_nums=10, saveval='min')

    ## start training
    for epoch in range(Epochs):
        if epoch > Stop_epoch:
            break
        for i, (img_id, sex, age, gsage, img) in enumerate(train_loader): #(img_id, gender, age, img)
            inputData = img.cuda()
            target_sex = sex.cuda()  # (batch,3,num_classes)
            target_age = age.cuda()
            target_agegsm = gsage[0].cuda()
            with autocast():  # mixed precision
                output_age, output_agegsm = model(inputData,target_sex)  # sigmoid will be done in loss !
                output_age=output_age.float()
                output_agegsm=output_agegsm.float()

            loss_age = criterion_age(output_age.squeeze(1), target_age.float())
            loss_agegs = criterion_agegs(output_agegsm, target_agegsm.float())
            loss = loss_age  + 0.3*loss_agegs
            loss = loss.float()
            model.zero_grad()
            scaler.scale(loss).backward()
            # loss.backward()
            scaler.step(optimizer)
            scaler.update()
            # optimizer.step()
            ema.update(model)

        scheduler.step()
        # store information

        print(args.data_name)
        print('Epoch [{}/{}],  LR {:.2e}, Loss: {:.6f}, loss_age:{:.6f}'
                .format(epoch, Epochs, scheduler.get_last_lr()[0], loss.item(), loss_age.item()))
        logger.info('Epoch [{}/{}], LR {:.2e}, Loss: {:.6f}, loss_age:{:.6f}'
                .format(epoch, Epochs,  scheduler.get_last_lr()[0], loss.item(), loss_age.item()))
        print("validate testing")
        model.eval()
        actaMAE, acta_ema_MAE = validate(val_loader, model, ema, args, logger, modeltype="m{}".format(epoch))
        model.train()
        savemodelname = savemodelinfo.updata(actaMAE, args.model_name, epoch)
        if len(savemodelname)>0:
            try:
                state_checkpoint = {
                    'state_dict': model.state_dict(),
                    'optimizer':optimizer.state_dict()}
                torch.save(state_checkpoint, os.path.join(savemodel_dir, savemodelname))
            except:
                pass

def validate(val_loader, model, ema_model, args, logger, modeltype="mx"):
    print("{} starting validation".format(args.model_name))
    saved_data=[]
    Eval_MAE = EvaluationMAE(age_names_label_dict[args.data_name])
    Eval_ema_MAE = EvaluationMAE(age_names_label_dict[args.data_name])
    for i, (img_id, gender, age, gsage, img) in enumerate(val_loader): 
        target_sex = gender.cuda()
        target_age = age.cuda()
        input = img.cuda()
        #target = target.max(dim=1)[0]
        # compute output
        with torch.no_grad():
            with autocast():
                output_age, output_agegsm = model(input, target_sex)
                ema_age, ema_agegsm = ema_model.module(input,target_sex)

                output_agew = model.weightage(output_age, output_agegsm)
                output_ema_agew = model.weightage(ema_age, ema_agegsm)
                # age
                output_regular_age = output_age.cpu()
                output_regular_ema_age = ema_age.cpu()
                # agew=age+gs
                output_regular_agew = output_agew.cpu()
                output_regular_ema_agew = output_ema_agew.cpu()

        Eval_MAE.update_mae(output_regular_agew.cpu(), target_age.cpu())
        Eval_ema_MAE.update_mae(output_regular_ema_agew.cpu(), target_age.cpu())

    table_info_age, actaMAE = Eval_MAE.metrics_table()
    table_info_ema_age, acta_ema_MAE = Eval_ema_MAE.metrics_table()
    logger.info(table_info_age)
    print(args.data_name)
    print(table_info_age)
    print(table_info_ema_age)
    return actaMAE, acta_ema_MAE


if __name__ == '__main__':
    main()