import argparse
import datetime
import os
from pathlib import Path
import time
import yaml
import yamlloader
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch import optim
from torch.nn import functional as F
from torch.nn import BCELoss

from network import UNet, UNet_Pretrained
from data_loader import Cephalometric_IMA
from mylogger import get_mylogger, set_logger_dir
from myTest import Tester
import matplotlib.pyplot as plt
import numpy as np

def get_noise_init(norm_type, noise_norm, init_norm, X):
    noise_init=2*torch.rand_like(X)-1
    noise_init=noise_init.view(X.size(0),-1)
    if isinstance(init_norm, torch.Tensor):
        init_norm=init_norm.view(X.size(0), -1)
    noise_init=init_norm*noise_init
    noise_init=noise_init.view(X.size())
    clip_norm_(noise_init, norm_type, init_norm)
    clip_norm_(noise_init, norm_type, noise_norm)
    return noise_init

def normalize_grad_(x_grad, norm_type, eps=1e-8):
    #x_grad is modified in place
    #x_grad.size(0) is batch_size
    with torch.no_grad():
        if norm_type == np.inf or norm_type == 'Linf':
            x_grad-=x_grad-x_grad.sign()
        elif norm_type == 2 or norm_type == 'L2':
            g=x_grad.view(x_grad.size(0), -1)
            l2_norm=torch.sqrt(torch.sum(g**2, dim=1, keepdim=True))
            l2_norm = torch.max(l2_norm, torch.tensor(eps, dtype=l2_norm.dtype, device=l2_norm.device))
            g *= 1/l2_norm
        else:
            raise NotImplementedError("not implemented.")
    return x_grad

def get_pgd_loss_fn_by_name(loss_fn):
    if loss_fn is None:
        loss_fn=torch.nn.CrossEntropyLoss(reduction="sum")
    elif isinstance(loss_fn, str):
        if loss_fn == 'none' or loss_fn == 'ce':
            loss_fn=torch.nn.CrossEntropyLoss(reduction="sum")
        elif loss_fn == 'bce':
            #loss_fn=torch.nn.BCEWithLogitsLoss(reduction="sum")
            loss_fn=binary_cross_entropy_with_logits
        elif loss_fn =='logit_margin_loss_binary' or loss_fn == 'lmb':
            loss_fn=logit_margin_loss_binary
        elif loss_fn == 'logit_margin_loss' or loss_fn == 'lm':
            loss_fn=logit_margin_loss
        elif loss_fn == 'soft_logit_margin_loss' or loss_fn == 'slm':
            loss_fn=soft_logit_margin_loss
        else:
            raise NotImplementedError("not implemented.")
    return loss_fn
#%%
def clip_norm_(noise, norm_type, norm_max):
    if not isinstance(norm_max, torch.Tensor):
        clip_normA_(noise, norm_type, norm_max)
    else:
        clip_normB_(noise, norm_type, norm_max)
#%%
def clip_normA_(noise, norm_type, norm_max):
    # noise is a tensor modified in place, noise.size(0) is batch_size
    # norm_type can be np.inf, 1 or 2, or p
    # norm_max is noise level
    if noise.size(0) == 0:
        return noise
    with torch.no_grad():
        if norm_type == np.inf or norm_type == 'Linf':
            noise.clamp_(-norm_max, norm_max)
        elif norm_type == 2 or norm_type == 'L2':
            N=noise.view(noise.size(0), -1)
            l2_norm= torch.sqrt(torch.sum(N**2, dim=1, keepdim=True))
            temp = (l2_norm > norm_max).squeeze()
            if temp.sum() > 0:
                N[temp]*=norm_max/l2_norm[temp]
        else:
            raise NotImplementedError("other norm clip is not implemented.")
    #-----------
    return noise
#%%
def clip_normB_(noise, norm_type, norm_max):
    # noise is a tensor modified in place, noise.size(0) is batch_size
    # norm_type can be np.inf, 1 or 2, or p
    # norm_max[k] is noise level for every noise[k]
    if noise.size(0) == 0:
        return noise
    with torch.no_grad():
        if norm_type == np.inf or norm_type == 'Linf':
            #for k in range(noise.size(0)):
            #    noise[k].clamp_(-norm_max[k], norm_max[k])
            N=noise.view(noise.size(0), -1)
            norm_max=norm_max.view(norm_max.size(0), -1)
            N=torch.max(torch.min(N, norm_max), -norm_max)
            N=N.view(noise.size())
            noise-=noise-N
        elif norm_type == 2 or norm_type == 'L2':
            N=noise.view(noise.size(0), -1)
            l2_norm= torch.sqrt(torch.sum(N**2, dim=1, keepdim=True))
            norm_max=norm_max.view(norm_max.size(0), 1)
            #print(l2_norm.shape, norm_max.shape)
            temp = (l2_norm > norm_max).squeeze()
            if temp.sum() > 0:
                norm_max=norm_max[temp]
                norm_max=norm_max.view(norm_max.size(0), -1)
                N[temp]*=norm_max/l2_norm[temp]
        else:
            raise NotImplementedError("not implemented.")
        #-----------
    return noise
#%%

def pgd_attack(model, X, Y, noise_norm, norm_type, max_iter, step,
               rand_init_norm=None, rand_init_Xn=None,
               targeted=False, clip_X_min=0, clip_X_max=1,
               refine_Xn_max_iter=10,
               Xn1_equal_X=False,
               Xn2_equal_Xn=False,
               stop_near_boundary=False,
               stop_if_label_change=False,
               stop_if_label_change_next_step=False,
               use_optimizer=False,
               loss_fn=None,
               model_eval_attack=False):
    #only apply pgd_attack to correctly classified samples
    #-------------------------------------------
    loss_fn=get_pgd_loss_fn_by_name(loss_fn)
    #-------------------------------------------
    train_mode=model.training# record the mode
    if model_eval_attack == True and train_mode == True:
        model.eval()#set model to evaluation mode
    #-----------------
    X = X.detach()
    #-----------------
    advc=torch.zeros(X.size(0), dtype=torch.int64, device=X.device)
    #-----------------
    if rand_init_norm is not None:
        noise_init=get_noise_init(norm_type, noise_norm, rand_init_norm, X)
        Xn = X + noise_init
    elif rand_init_Xn is not None:
        Xn = rand_init_Xn.clone().detach()
    else:
        raise ValueError('invalid input')
    #-----------------
    Xn1=X.detach().clone() # about to across decision boundary
    Xn2=X.detach().clone() # just across decision boundary
    Ypn_old=Y # X is correctly classified
    #-----------------
    noise=(Xn-X).detach()
    if use_optimizer == True:
        optimizer = optim.Adamax([noise], lr=step)
    #-----------------
    for n in range(0, max_iter+1):
        Xn = Xn.detach()
        Xn.requires_grad = True
        Zn, Ypn=run_model(model, Xn)
        loss = loss_fn(Zn, Y)
        Ypn_e_Y=(Ypn==Y)
        Ypn_ne_Y=(Ypn!=Y)
        Ypn_old_e_Y=(Ypn_old==Y)
        Ypn_old_ne_Y=(Ypn_old!=Y)
        #---------------------------
        #targeted attack, Y should be filled with targeted class label
        if targeted == False:
            A=Ypn_e_Y
            A_old=Ypn_old_e_Y
            B=Ypn_ne_Y
        else:
            A=Ypn_ne_Y
            A_old=Ypn_old_ne_Y
            B=Ypn_e_Y
            loss=-loss
        #---------------------------
        temp1=(A&A_old)&(advc<1)
        Xn1[temp1]=Xn[temp1].data
        temp2=(B&A_old)&(advc<1)
        Xn2[temp1]=Xn[temp1].data
        Xn2[temp2]=Xn[temp2].data
        advc[B]+=1
        #---------------------------
        if n < max_iter:
            #loss.backward() will update W.grad
            grad_n=torch.autograd.grad(loss, Xn)[0]
            grad_n=normalize_grad_(grad_n, norm_type)
            if use_optimizer == True:
                noise.grad=-grad_n #grad ascent to maximize loss
                optimizer.step()
            else:
                Xnew = Xn + step*grad_n
                noise = Xnew-X
            #---------------------
            clip_norm_(noise, norm_type, noise_norm)
            Xn = torch.clamp(X+noise, clip_X_min, clip_X_max)
            noise.data -= noise.data-(Xn-X).data
            Ypn_old=Ypn
        #---------------------------
        #if n==0 or n ==max_iter:
        #    loss_sum=logit_margin_loss(Zn, Y, reduction='sum')
        #    print(loss.item(), loss_sum.item())
    #print('pgd is done...................')
    #---------------------------
    #Zn1 = model(Xn1)
    #Ypn1 = Zn1.data.max(dim=1)[1]
    #print('Ypn1', (Ypn1==Y).sum().item(), Y.size(0))
    #---------------------------
    Xn_out = Xn.detach()
    if Xn1_equal_X:
        Xn1=X.detach().clone()
    if Xn2_equal_Xn:
        Xn2=Xn
    if stop_near_boundary == True:
        temp=advc>0
        if temp.sum()>0:
            Xn_out[temp]=refine_Xn_onto_boundary(model, Xn1[temp], Xn2[temp], Y[temp], refine_Xn_max_iter)
    elif stop_if_label_change == True:
        temp=advc>0
        if temp.sum()>0:
            Xn_out[temp]=refine_Xn2_onto_boundary(model, Xn1[temp], Xn2[temp], Y[temp], refine_Xn_max_iter)
    elif stop_if_label_change_next_step == True:
        temp=advc>0
        if temp.sum()>0:
            Xn_out[temp]=refine_Xn1_onto_boundary(model, Xn1[temp], Xn2[temp], Y[temp], refine_Xn_max_iter)
    #---------------------------
    if train_mode == True and model.training == False:
        model.train()
    #---------------------------
    return Xn_out, advc

def refine_onto_boundary(model, Xn1, Xn2, Y, max_iter):
#note: Xn1 and Xn2 will be modified
    with torch.no_grad():
        Xn=(Xn1+Xn2)/2
        for k in range(0, max_iter):
            Zn, Ypn=run_model(model, Xn)
            Ypn_e_Y=Ypn==Y
            Ypn_ne_Y=Ypn!=Y
            Xn1[Ypn_e_Y]=Xn[Ypn_e_Y]
            Xn2[Ypn_ne_Y]=Xn[Ypn_ne_Y]
            Xn=(Xn1+Xn2)/2
            #if k==0 or k ==max_iter-1:
            #    loss=logit_margin_loss(Zn, Y, reduction='sum_abs')
            #    print(loss.item())
        #print('refine done')
    return Xn, Xn1, Xn2
#%%
def refine_Xn_onto_boundary(model, Xn1, Xn2, Y, max_iter):
#note: Xn1 and Xn2 will be modified
    Xn, Xn1, Xn2=refine_onto_boundary(model, Xn1, Xn2, Y, max_iter)
    return Xn
#%%
def refine_Xn1_onto_boundary(model, Xn1, Xn2, Y, max_iter):
#note: Xn1 and Xn2 will be modified
    Xn, Xn1, Xn2=refine_onto_boundary(model, Xn1, Xn2, Y, max_iter)
    return Xn1
#%%
def refine_Xn2_onto_boundary(model, Xn1, Xn2, Y, max_iter):
#note: Xn1 and Xn2 will be modified
    Xn, Xn1, Xn2=refine_onto_boundary(model, Xn1, Xn2, Y, max_iter)
    return Xn2
#%%

def repeated_pgd_attack(model, X, Y, noise_norm, norm_type, max_iter, step,
                        rand_init_norm=None, rand_init_Xn=None,
                        targeted=False, clip_X_min=0, clip_X_max=1,
                        refine_Xn_max_iter=10,
                        Xn1_equal_X=False,
                        Xn2_equal_Xn=False,
                        stop_near_boundary=False,
                        stop_if_label_change=False,
                        stop_if_label_change_next_step=False,
                        use_optimizer=False,
                        loss_fn=None,
                        model_eval_attack=False,
                        num_repeats=1):
    for m in range(0, num_repeats):
        Xm, advcm = pgd_attack(model, X, Y, noise_norm, norm_type, max_iter, step,
                               rand_init_norm, rand_init_Xn,
                               targeted, clip_X_min, clip_X_max,
                               refine_Xn_max_iter,
                               Xn1_equal_X,
                               Xn2_equal_Xn,
                               stop_near_boundary,
                               stop_if_label_change,
                               stop_if_label_change_next_step,
                               use_optimizer,
                               loss_fn,
                               model_eval_attack)
        if m == 0:
            Xn=Xm
            advc=advcm
        else:
            temp=advcm>0
            advc[temp]=advcm[temp]
            Xn[temp]=Xm[temp]
    #--------
    return Xn, advc

def get_loss_function(z_size, reduction='sum'):
    if len(z_size) <=1:
        return torch.nn.BCEWithLogitsLoss(reduction=reduction)
    else:
        return torch.nn.CrossEntropyLoss(reduction=reduction)

def run_model(model, X):
    Z=model(X)
    if len(Z.size()) <= 1:
        Yp = (Z.data>0).to(torch.int64)
    else:
        Yp = Z.data.max(dim=1)[1]
    return Z, Yp

def L1Loss(pred, gt, mask=None,reduction = "mean"):
    # L1 Loss for offset map
    assert(pred.shape == gt.shape)
    gap = pred - gt
    distence = gap.abs()
    if mask is not None:
        # Caculate grad of the area under mask
        distence = distence * mask
        
    if reduction =="mean":
        # sum in this function means 'mean'
        return distence.sum() / mask.sum()
        # return distence.mean()
    else:
        return distence
    

def focal_loss(pred, gt):
    return (-(1-pred)*gt*torch.log(pred) - pred*(1-gt)*torch.log(1-pred)).mean()

def total_loss(heatmap, guassian_mask, regression_y, offset_y, regression_x, offset_x, mask, lamb=2, reduction = 'sum'):
    # loss
    if reduction == 'sum':
        loss_logic_fn = BCELoss()
        loss_regression_fn = L1Loss
        # the loss for heatmap
        logic_loss = loss_logic_fn(heatmap, guassian_mask)
        # the loss for offset
        regression_loss_y = loss_regression_fn(regression_y, offset_y, mask, reduction = "mean")
        regression_loss_x = loss_regression_fn(regression_x, offset_x, mask, reduction = "mean")
        return  regression_loss_x + regression_loss_y + logic_loss * lamb, regression_loss_x + regression_loss_y
    else: 
        # every sample has its loss, none reduction
        loss_logic_fn = BCELoss(reduction = reduction)
        loss_regression_fn = L1Loss
        # the loss for heatmap
        logic_loss = loss_logic_fn(heatmap, guassian_mask)
        # the loss for offset
        regression_loss_y = loss_regression_fn(regression_y, offset_y, mask, reduction = "none")
        regression_loss_x = loss_regression_fn(regression_x, offset_x, mask, reduction = "none")
        return  regression_loss_x + regression_loss_y + logic_loss * lamb, regression_loss_x + regression_loss_y
  
    

def run_model_std_reg(net, img, mask, offset_y, offset_x, guassian_mask, return_loss=False, reduction='none'):
    heatmap, regression_y, regression_x = net(img)
    
    if return_loss == True:
        loss=total_loss(heatmap, guassian_mask, regression_y, offset_y, regression_x, offset_x, mask, reduction = reduction)       
        return heatmap, regression_y, regression_x, loss
    else:
        return heatmap, regression_y, regression_x
#
def run_model_adv_reg(net, img, mask, offset_y, offset_x, guassian_mask, return_loss=False, reduction='none'):
    heatmap, regression_y, regression_x = net(img)
    
    if return_loss == True:
        loss=total_loss(heatmap, guassian_mask, regression_y, offset_y, regression_x, offset_x, mask, reduction = reduction)       
        return heatmap, regression_y, regression_x, loss
    else:
        return heatmap, regression_y, regression_x
#
def classify_model_std_output_reg(heatmap, guassian_mask, regression_y, offset_y, regression_x, offset_x, mask):
    threshold=1
    loss= total_loss(heatmap, guassian_mask, regression_y, offset_y, regression_x, offset_x, mask, lamb=2, reduction = 'none')
    Yp_e_Y=(loss<=threshold)
    return Yp_e_Y
#
def classify_model_adv_output_reg(heatmap, guassian_mask, regression_y, offset_y, regression_x, offset_x, mask):
    threshold=0.1
    loss= total_loss(heatmap, guassian_mask, regression_y, offset_y, regression_x, offset_x, mask, lamb=2, reduction = 'none')
    Yp_e_Y=(loss<=threshold)
    return Yp_e_Y

def IMA_loss(net, img, mask, offset_y, offset_x, guassian_mask, 
             margin, norm_type, max_iter, step,
             rand_init_norm=None, rand_init_Xn=None,
             clip_X_min=0, clip_X_max=1,
             refine_Xn_max_iter=10,
             Xn1_equal_X=False,
             Xn2_equal_Xn=False,
             stop_near_boundary=True,
             stop_if_label_change=False,
             stop_if_label_change_next_step=False,
             beta=0.5, beta_position=1,
             use_optimizer = False,
             pgd_num_repeats=1,
             run_model_std=None, classify_model_std_output=None,
             run_model_adv=None, classify_model_adv_output=None
             ):
    #----------------------------------
    if isinstance(step, torch.Tensor):
        temp=tuple([1]*len(X[0].size()))
        step=step.view(-1, *temp)
    #-----------------------------------
    heatmap, regression_y, regression_x, loss_X=run_model_std(net, img, mask, offset_y, offset_x, guassian_mask,return_loss=True)
    Yp_e_Y=classify_model_std_output(heatmap, guassian_mask, regression_y, offset_y, regression_x, offset_x, mask)
    Yp_ne_Y=~Yp_e_Y 
    #-----------------------------------
    loss1=torch.tensor(0.0, dtype=img.dtype, device=img.device, requires_grad=True)
    loss2=torch.tensor(0.0, dtype=img.dtype, device=img.device, requires_grad=True)
    loss3=torch.tensor(0.0, dtype=img.dtype, device=img.device, requires_grad=True)
    Xn=torch.tensor([], dtype=img.dtype, device=img.device)
    Ypn=torch.tensor([], dtype=guassian_mask.dtype, device=guassian_mask.device)
    advc=torch.zeros(img.size(0), dtype=torch.int64, device=img.device)
    idx_n=torch.tensor([], dtype=torch.int64, device=img.device)
    #----------------------------------
    if Yp_ne_Y.sum().item()>0:
        loss1 = loss_X[Yp_ne_Y].sum()/img.size(0)
    if Yp_e_Y.sum().item()>0:
        loss2 = loss_X[Yp_e_Y].sum()/img.size(0)
    #---------------------------------
    train_mode=net.training# record the mode
    #---------------------------------
    # we ingore the re_initial, there is no need to use enable_loss3
    """
    enable_loss3=False
    if Yp_e_Y.sum().item()>0 and beta>0:
         enable_loss3=True
    """
    #----------------------------------
    #if enable_loss3 == True:
    Xn, advc[Yp_e_Y] = repeated_pgd_attack(net, img, mask, offset_y, offset_x, guassian_mask, 
                                           noise_norm=margin, norm_type=norm_type,
                                           max_iter=max_iter, step=step,
                                           rand_init_norm=rand_init_norm, rand_init_Xn=rand_init_Xn,
                                           clip_X_min=clip_X_min, clip_X_max=clip_X_max,
                                           refine_Xn_max_iter=refine_Xn_max_iter,
                                           Xn1_equal_X=Xn1_equal_X,
                                           Xn2_equal_Xn=Xn2_equal_Xn,
                                           stop_near_boundary=stop_near_boundary,
                                           stop_if_label_change=stop_if_label_change,
                                           stop_if_label_change_next_step=stop_if_label_change_next_step,
                                           use_optimizer=use_optimizer,
                                           run_model=run_model_adv, classify_model_output=classify_model_adv_output,
                                           num_repeats=pgd_num_repeats)
    #--------------------------------------------

    if train_mode == True and net.training == False:
        net.train()
    #--------------------------------------------
    idx_n=torch.arange(0,img.size(0))[Yp_e_Y]
    h_n, reg_y_n, reg_x_n, loss_Xn=run_model_std(net, Xn, mask, offset_y, offset_x, guassian_mask,return_loss=True)
    Xn=Xn[idx_n]
    if idx_n.size(0)>0:   
        loss3 = loss_Xn[idx_n].sum()/Xn.size(0)
    #--------------------------------------------
    if beta_position == 0:
        loss=(1-beta)*loss1+(beta*0.5)*(loss2+loss3)
    elif beta_position == 1:
        loss=(1-beta)*(loss1+loss2)+beta*loss3
    elif beta_position == 2:
        loss=loss1+(1-beta)*loss2+beta*loss3
    elif beta_position == 3:
        loss=(1-beta)*loss1+beta*loss3
    else:
        raise ValueError('unknown beta_position')
    #--------------------------------------------
    if train_mode == True and net.training == False:
        net.train()
    #--------------------------------------------
    return loss, heatmap, regression_y, regression_x, advc, Xn, Ypn, idx_n    

def IMA_update_margin(E, delta, max_margin, flag1, flag2, margin_new):
    # margin: to be updated
    # delta: margin expansion step size
    # max_margin: maximum margin
    # flag1, flag2, margin_new: from IMA_check_margin
    expand=(flag1==1)&(flag2==1)
    no_expand=(flag1==0)&(flag2==1)
    args.E[expand]+=delta
    args.E[no_expand]=margin_new[no_expand]
    #when wrongly classified, do not re-initialize
    #args.E[flag2==0]=delta
    E.clamp_(min=0, max=max_margin)

if __name__ == "__main__":
    #CUDA_VISIBLE_DEVICES=0
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    #device = torch.device('cuda:1')
    # Parse command line options
    parser = argparse.ArgumentParser(description="Train Unet landmark detection network")
    parser.add_argument("--tag", default='train', help="name of the run")
    parser.add_argument("--config_file", default="config.yaml", help="default configs")
    args = parser.parse_args()
 
    # Load yaml config file
    with open(args.config_file) as f:
        config = yaml.load(f, Loader=yamlloader.ordereddict.CLoader)
    
    # Create Logger
    logger = get_mylogger()
    logger.info(config)

    # Create runs dir
    tag = str(datetime.datetime.now()).replace(' ', '_') if args.tag == '' else args.tag
    runs_dir = "./runs/" + tag
    runs_path = Path(runs_dir)
    config['runs_dir'] = runs_dir
    if not runs_path.exists():
        runs_path.mkdir()
    #set_logger_dir(logger, runs_dir)

    dataset = Cephalometric_IMA(config['dataset_pth'], 'Train')
    dataloader = DataLoader(dataset, batch_size=config['batch_size'],
                                shuffle=True, num_workers=config['num_workers'])
    
    # net = UNet(3, config['num_landmarks'])
    net = UNet_Pretrained(3, config['num_landmarks'])
    net = torch.nn.DataParallel(net)
    net = net.cuda()
    logger.info(net)

    optimizer = optim.Adam(params=net.parameters(), \
        lr=config['learning_rate'], betas=(0.9,0.999), eps=1e-08, weight_decay=1e-4)
    
    scheduler = StepLR(optimizer, config['decay_step'], gamma=config['decay_gamma'])


    # Tester
    tester = Tester(logger, config, tag=args.tag)
    
    # parameters
    stop = 1
    stop_near_boundary=False
    stop_if_label_change=False
    stop_if_label_change_next_step=False
    if stop==1:
        stop_near_boundary=True
    elif stop==2:
        stop_if_label_change=True
    elif stop==3:
        stop_if_label_change_next_step=True  
    #======================
    
    sample_count_train = 150
    noise = 0.1
    epoch_refine = config['num_epochs']
    delta = noise/epoch_refine
    E = delta*torch.ones(sample_count_train, dtype=torch.float32)
    alpha = 4    
    max_iter=20    
    #======================
    
    loss_train_list = list()
    loss_val_list = list()
    MRE_list = list()
    
    for epoch in range(config['num_epochs']):
        logic_loss_list = list()
        regression_loss_list = list()
        flag1=torch.zeros(len(args.E), dtype=torch.float32)
        flag2=torch.zeros(len(args.E), dtype=torch.float32)
        E_new=args.E.detach().clone()
        net.train()
        for img, mask, guassian_mask, offset_y, offset_x, landmark_list, idx in tqdm(dataloader):
            img, mask, offset_y, offset_x, guassian_mask = img.cuda(), mask.cuda(), \
                offset_y.cuda(), offset_x.cuda(), guassian_mask.cuda()
                
            #heatmap, regression_y, regression_x = net(img)
            #loss, regLoss  = total_loss(heatmap, guassian_mask, regression_y, offset_y, regression_x, offset_x, mask, config['lambda'])
            net.zero_grad()
            #......................................................................................................
            rand_init_norm=torch.clamp(E[idx]-delta, min=delta).cuda()
            margin=E[idx].cuda()
            step=alpha*margin/max_iter
            #......................................................................................................
            loss, heatmap, regression_y, regression_x, advc, Xn ,Ypn, idx_n = IMA_loss(net, img, mask, offset_y, offset_x, guassian_mask,
                                                    norm_type= np.inf,
                                                    rand_init_norm=rand_init_norm,
                                                    margin=margin,
                                                    max_iter=max_iter,
                                                    step=step,
                                                    refine_Xn_max_iter=10,
                                                    Xn1_equal_X=0,
                                                    Xn2_equal_Xn=0,
                                                    stop_near_boundary=stop_near_boundary,
                                                    stop_if_label_change=stop_if_label_change,
                                                    stop_if_label_change_next_step=stop_if_label_change_next_step,
                                                    beta=0.5, beta_position=1,
                                                    use_optimizer=False,                                                
                                                    run_model_std=run_model_std_reg,
                                                    classify_model_std_output=classify_model_std_output_reg,
                                                    run_model_adv=run_model_adv_reg,
                                                    classify_model_adv_output=classify_model_adv_output_reg)
            

            loss.backward()
            optimizer.step()
         #--------------------update the margins
        #Yp_e_Y=classify_model_std_output_seg(Yp, target)
        flag1[idx[advc==0]]=1
        #flag2[idx[Yp_e_Y]]=1
        flag2[idx]=1
        if idx_n.shape[0]>0:
            temp=torch.norm((Xn-img[idx_n]).view(Xn.shape[0], -1), p=args.norm_type, dim=1).cpu()
            #E_new[idx[idx_n]]=torch.min(E_new[idx[idx_n]], temp)     
            #bottom = args.delta*torch.ones(E_new.size(0), dtype=E_new.dtype, device=E_new.device)
            E_new[idx[idx_n]] = (E_new[idx[idx_n]]+temp)/2# use mean to refine the margin to reduce the effect of augmentation on margins
        #-----------------------------------------------------------------------
        IMA_update_margin(E, delta, noise, flag1, flag2, E_new) 
        loss_train = sum(loss) / dataset.__len__()
        loss_train_list.append(loss_train)
        logger.info("Epoch {} Training  logic loss {}".format(epoch, loss_train))
        
        
        #validation part       
        MRE, loss_val,loss_logic,  loss_reg = tester.validate(net)
        logger.info("Epoch {} Testing MRE {},  loss {}, logic loss {}, reg loss {}".format(epoch, MRE, loss_val, loss_logic, loss_reg))
        loss_val_list.append(loss_val)
        MRE_list.append(MRE)
        
        
        # save model and plot the trend
        if (epoch + 1) % config['save_seq'] == 0:
            logger.info(runs_dir + "/model_epoch_{}.pth".format(epoch))
            torch.save(net.state_dict(), runs_dir + "/model_epoch_{}.pth".format(epoch))
            # plot the trend
            cols = ['b','g','r','y','k','m','c']
            
            fig,axs = plt.subplots(1,3, figsize=(15,5))

            #ax = fig.add_subplot(111)
            X = list(range(epoch+1))
            axs[0].plot(X, loss_train_list, color=cols[0], label="Training Loss")
            axs[1].plot(X, loss_val_list, color=cols[1], label="Validation Loss")
            axs[2].plot(X, MRE_list, color=cols[2], label="MRE")
            axs[0].set_xlabel("epoch")   
            axs[1].set_xlabel("epoch")
            axs[2].set_xlabel("epoch")
            axs[0].legend()
            axs[1].legend()
            axs[2].legend()           
            fig.savefig(runs_dir +"training.png")
            #save the last epoch
            config['last_epoch'] = epoch

        # dump yaml
        with open(runs_dir + "/config.yaml", "w") as f:
            yaml.dump(config, f)

    # # Test
    # tester.test(net)
