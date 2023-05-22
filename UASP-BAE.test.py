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
from srcfunc.datasets.toothopgdata import toothopg, GussiCut, age_names_label_dict
from srcfunc.metrics.EvaluationMAE import EvaluationMAE
from srcfunc.models import AgeJoiner, create_model, prepare_device
from srcfunc.loss_functions.losses import  ACLossGs
from srcfunc.helper_functions.helper_functions import ModelEma, savetenmodel, add_weight_decay
from srcfunc.helper_functions.logging import setup_logging
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

    # Setup model
    print('{} creating model...'.format(args.model_name))
    backbone = create_model(args).cuda()

    # get backbone without fc
    backbone.forward = backbone.forward_features

    #del backbone.avgpool
    num_features = backbone.num_features
    del backbone.head
    #del backbone.fc
    model = AgeJoiner(backbone, n_channels=num_features, agegsch=int((agerange[1]-agerange[0])*2)+1 ).cuda()
    print('done\n') 
    # Pytorch Data loader
    imgs_dir, label_csv = "./datasets/test","./datasets/test.%s.csv"%(args.data_name)
    #ToothCls_val = toothopg(imgs_dir, label_csv, imagesize=(args.image_size, args.image_size), tempmemary=False)
    ToothCls_test = toothopg(imgs_dir, label_csv, imagesize=args.image_size, tempmemary=False, agerange=agerange,
                             mapnum=int((agerange[1]-agerange[0])*2)+1,istest=True)
    test_loader = torch.utils.data.DataLoader(
        ToothCls_test, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    # load model params
    if args.model_path is not None:
        checkpoint = torch.load(args.model_path, map_location=torch.device(device))
        #filtered_dict = {k: v for k, v in checkpoint["state_dict"]['model'].items() if
        #                 (k in model.state_dict() and 'head.fc' not in k)}
        model.load_state_dict(checkpoint, strict=False)
    model.eval()    

    # set optimizer
    print("{} start testing".format(args.model_name))
    saved_data=[]
    Eval_MAE = EvaluationMAE(age_names_label_dict[args.data_name])
    Eval_ema_MAE = EvaluationMAE(age_names_label_dict[args.data_name])
    for i, (img_id, gender, age, gsage, img) in enumerate(test_loader): 
        target_sex = gender.cuda()
        target_age = age.cuda()
        input = img.cuda()
        #target = target.max(dim=1)[0]
        # compute output
        with torch.no_grad():
            with autocast():
                output_age, output_agegsm = model(input, target_sex)
                output_agew = model.weightage(output_age, output_agegsm)
                # age
                output_regular_age = output_age.cpu()
                # agew=age+gs
                output_regular_agew = output_agew.cpu()

        Eval_MAE.update_mae(output_regular_agew.cpu(), target_age.cpu())
        
        for idx, img_name in enumerate(img_id):
            predinfo={}
            predinfo = {"img_id":img_id[idx],
                            "sex_gt":target_sex.detach().cpu().numpy()[idx],
                            "age_gt":target_age.detach().cpu().numpy()[idx],
                            "age_test":output_regular_age.squeeze(1).detach().cpu().numpy()[idx],
                            "age_testw":output_regular_agew.squeeze(1).detach().cpu().numpy()[idx]
                            }
            saved_data.append(predinfo)
            print("img_id: %s"%predinfo['img_id'], 
                  "sex_gt= %d"%predinfo['sex_gt'],
                  "age_gt= %0.04f"%predinfo['age_gt'],
                  "age_test= %0.04f"%predinfo['age_test'],
                  "age_testw= %0.04f"%predinfo['age_testw']
                  )

    ## save to txt
    saved_name = 'testing.{}_{}.csv'.format(args.model_name, run_id)
    save_dir=Path("./outputs/test/{}_{}_{}".format(args.data_name, args.model_name,run_id))
    save_dir.mkdir(parents=True, exist_ok=True)

    #np.savetxt(os.path.join(save_dir, saved_name), saved_data)
    csvtestdata = pd.DataFrame(saved_data)
    csvtestdata.to_csv(os.path.join(save_dir, saved_name))

    table_info_age, actaMAE = Eval_MAE.metrics_table()
    
    Eval_MAE.boxplots(os.path.join(save_dir, saved_name))
    Eval_MAE.violinplots(os.path.join(save_dir, saved_name))

    print(args.data_name)
    print(table_info_age)


if __name__ == '__main__':
    main()