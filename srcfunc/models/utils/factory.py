# -*- coding: utf-8 -*-
"""
Created by @author: Carlson Zhang. Xi'an Jiaotong University.
"""
import logging
import torch
from torch import nn
logger = logging.getLogger(__name__)
from ..tresnet import TResnetM, TResnetL, TResnetXL
from ..ttresnet import TTResnetM, TTResnetL, TTResnetXL
from ..res2net import res2net50, res2next50 
from ..coatnet import coatnet_0, coatnet_1, coatnet_2, coatnet_3, coatnet_4
from ..creatvit import m_Vit, m_T2TViT, m_CrossViT
from ..cmt import cmt_b, cmt_s, cmt_xs, cmt_ti
from ..csatnet import UASP_BAE_net0, UASP_BAE_net1, UASP_BAE_net2,UASP_BAE_net3

def prepare_device(n_gpu_use):
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:    
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

def create_model(args):
    """Create a model
    """
    model_params = {'args': args, 'num_classes': args.num_classes, 'image_size': args.image_size}
    args = model_params['args']
    args.model_name = args.model_name.lower()
    print("{} model Creating".format(args.model_name))
    logger.info("{} model Creating".format(args.model_name))
    if args.model_name=='tresnet_m':
        model = TResnetM(model_params)
    elif args.model_name=='tresnet_l':
        model = TResnetL(model_params)
    elif args.model_name=='tresnet_xl':
        model = TResnetXL(model_params)

    elif args.model_name=='ttresnet_m':
        model = TTResnetL(model_params)
    elif args.model_name=='ttresnet_l':
        model = TTResnetL(model_params)
    elif args.model_name=='ttresnet_xl':
        model = TTResnetXL(model_params)
        
    elif args.model_name=='res2net50':
        model = res2net50(model_params)
    elif args.model_name=='res2next50':
        model = res2next50(model_params)

    elif args.model_name=='coatnet_0':
        model = coatnet_0(model_params)
    elif args.model_name=='coatnet_1':
        model = coatnet_1(model_params)
    elif args.model_name=='coatnet_2':
        model = coatnet_2(model_params)
    elif args.model_name=='coatnet_3':
        model = coatnet_3(model_params)
    elif args.model_name=='coatnet_4':
        model = coatnet_4(model_params)
        
    elif args.model_name=='m_vit':
        model = m_Vit(model_params)
    elif args.model_name=='m_t2tvit':
        model = m_T2TViT(model_params)
    elif args.model_name=='m_crossvit':
        model = m_CrossViT(model_params)

    elif args.model_name=='cmt_b':
        model = cmt_b(model_params)
    elif args.model_name=='cmt_s':
        model = cmt_s(model_params)
    elif args.model_name=='cmt_xs':
        model = cmt_xs(model_params)
    elif args.model_name=='cmt_ti':
        model = cmt_ti(model_params)

    elif args.model_name=='uasp_bae_net0':
        model = UASP_BAE_net0(model_params)
    elif args.model_name=='uasp_bae_net1':
        model = UASP_BAE_net1(model_params)
    elif args.model_name=='uasp_bae_net2':
        model = UASP_BAE_net2(model_params)
    elif args.model_name=='uasp_bae_net3':
        model = UASP_BAE_net3(model_params)

    else:
        print("model: {} not found !!".format(args.model_name))
        exit(-1)
    
    return model

class projection_MLP(nn.Module):
    def __init__(self, n_channels):
        super(projection_MLP, self).__init__()
        self.projection_head = nn.Sequential()
        self.projection_head.add_module('W1', nn.Linear(
            n_channels, n_channels))
        self.projection_head.add_module('ReLU', nn.ReLU())
        self.projection_head.add_module('W2', nn.Linear(
            n_channels, 128))
        
    def forward(self, x):
        return self.projection_head(x)

class AgeJoiner(nn.Sequential):
    def __init__(self, backbone, n_channels, agegsch = 42, args=None):
        super().__init__(backbone)
        self.agegsch = agegsch
        self.age = nn.Sequential()
        n_channels_age = n_channels + int(n_channels/10)
        self.age.add_module('W1', nn.Linear(n_channels_age, n_channels//2))
        self.age.add_module('LeakyReLU', nn.LeakyReLU())
        #self.age.add_module('W2', nn.Linear(n_channels//2, n_channels//4))
        #self.age.add_module('LeakyReLU', nn.LeakyReLU())
        self.age.add_module('W3', nn.Linear(n_channels//2, 1))

        self.age_gs = nn.Sequential()
        n_channels_gs= n_channels + int(n_channels/10)
        self.age_gs.add_module('W1', nn.Linear(n_channels_gs, n_channels//4))
        self.age_gs.add_module('LeakyReLU', nn.LeakyReLU())
        self.age_gs.add_module('W2', nn.Linear(n_channels//4, self.agegsch))
        self.age_gs.add_module('Sigmoid', nn.Sigmoid())

    def weightage(self, age_y, age_gs):
        gs = torch.arange(5,25,20/self.agegsch)
        agerange = gs.cuda(age_y.device)
        age_wgs = torch.mm(age_gs, agerange.unsqueeze(dim=0).T)/self.agegsch
        age_gs = 1/age_gs.sum(axis=1)
        return 0.8*age_y + 0.2*torch.mul(age_gs.unsqueeze(dim=1), age_wgs)
    
    def forward(self, rawdata, sex=None): ## sex = 0, 1-man, 2-female
        features = self[0](rawdata)
        if sex is not None:    
            ch, wh = features.shape
            sex[sex==2]=-1 ##
            sex_p = torch.ones(ch,int(wh/10),dtype=torch.float).cuda(features.device)
            pirisex = torch.mul(sex_p, sex.unsqueeze(dim=1))
            features = torch.cat([features,pirisex],dim=-1)   
        ageout = self.age(features)
        agegsm = self.age_gs(features)
        #weightage = self.weightage(ageout,agegsm)
        return ageout, agegsm