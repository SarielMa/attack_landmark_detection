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
from PGD import IMA_loss
from metric import total_loss, l1_matric

def run_model_std_reg(net, img, mask, offset_y, offset_x, guassian_mask, return_loss=False, reduction='none'):
    heatmap, regression_y, regression_x = net(img)
    
    if return_loss == True:
        loss, _=total_loss(heatmap, guassian_mask, regression_y, offset_y, regression_x, offset_x, mask, reduction = reduction)       
        return heatmap, regression_y, regression_x, loss
    else:
        return heatmap, regression_y, regression_x
#
def run_model_adv_reg(net, img, mask, offset_y, offset_x, guassian_mask, return_loss=False, reduction='none'):
    heatmap, regression_y, regression_x = net(img)
    
    if return_loss == True:
        loss, _=total_loss(heatmap, guassian_mask, regression_y, offset_y, regression_x, offset_x, mask, reduction = reduction)       
        return heatmap, regression_y, regression_x, loss
    else:
        return heatmap, regression_y, regression_x
#
def classify_model_std_output_reg(heatmap, guassian_mask, regression_y, offset_y, regression_x, offset_x, mask):
    threshold1=0
    threshold2=10000
    threshold3 = 10000
    r, ry, rx= l1_matric(heatmap, guassian_mask, regression_y, offset_y, regression_x, offset_x, mask)
    Yp_e_Y=(r>=threshold1) & (ry <=threshold2) & (rx <= threshold3)
    return Yp_e_Y
#
def classify_model_adv_output_reg(heatmap, guassian_mask, regression_y, offset_y, regression_x, offset_x, mask):
    #3Z
    threshold1=0
    # min
    #threshold1=0.9076504
    r, ry, rx= l1_matric(heatmap, guassian_mask, regression_y, offset_y, regression_x, offset_x, mask)
    Yp_e_Y=(r>=threshold1) 
    return Yp_e_Y


def IMA_update_margin(E, delta, max_margin, flag1, flag2, margin_new):
    # margin: to be updated
    # delta: margin expansion step size
    # max_margin: maximum margin
    # flag1, flag2, margin_new: from IMA_check_margin
    expand=(flag1==1)&(flag2==1)
    no_expand=(flag1==0)&(flag2==1)
    E[expand]+=delta
    E[no_expand]=margin_new[no_expand]
    #when wrongly classified, do not re-initialize
    #args.E[flag2==0]=delta
    E.clamp_(min=0, max=max_margin)
    print (expand.sum().item(),"samples are expanded.....")

if __name__ == "__main__":

    #device = torch.device('cuda:1')
    # Parse command line options
    parser = argparse.ArgumentParser(description="Train Unet landmark detection network")
    parser.add_argument("--tag", default='BPGD_10', help="name of the run")
    parser.add_argument("--cuda", default = '1')
    parser.add_argument("--config_file", default="config.yaml", help="default configs")
    parser.add_argument("--pretrain")
    args = parser.parse_args()
 
    #CUDA_VISIBLE_DEVICES=0
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=args.cuda
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
    
    # net = UNet(3, config['num_landmarks'])##################################################
    net = UNet_Pretrained(3, config['num_landmarks'])
    iteration  = config['num_epochs']-1
    # load the pretrained clean model
    if args.pretrain == "True":
        checkpoints = torch.load("./runs/base/model_epoch_{}.pth".format(iteration))
        newCP = dict()
        #adjust the keys(remove the "module.")
        for k in checkpoints.keys():
            newK = ""
            if "module." in k:
                newK = ".".join(k.split(".")[1:])
            else:
                newK = k
            newCP[newK] = checkpoints[k]
        net.load_state_dict(newCP)   
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
    noise = float(args.tag.split("_")[1])
    epoch_refine = config['num_epochs']
    #delta = 23*noise/epoch_refine
    delta  = 10
    updateRate = 0.75
    #delta = 1
    E = delta*torch.ones(sample_count_train, dtype=torch.float32)
    bottom = delta*torch.ones(sample_count_train, dtype=torch.float32)
    alpha = 5    
    max_iter=20   
    norm_type = 2
    #======================
    
    loss_train_list = list()
    loss_val_list = list()
    MRE_list = list()
    
    for epoch in range(config['num_epochs']):
        loss_list = list()
        regression_loss_list = list()
        flag1=torch.zeros(len(E), dtype=torch.float32)
        flag2=torch.zeros(len(E), dtype=torch.float32)
        E_new=E.detach().clone()
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
                                                    norm_type= norm_type,
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
            loss_list.append(loss)
         #--------------------update the margins-
            #Yp_e_Y=classify_model_std_output_seg(Yp, target)
            flag1[idx[advc==0]]=1
            #flag2[idx[Yp_e_Y]]=1
            flag2[idx]=1
            if idx_n.shape[0]>0:
                temp=torch.norm((Xn-img[idx_n]).view(Xn.shape[0], -1), p=norm_type, dim=1).cpu()
                #E_new[idx[idx_n]]=torch.min(E_new[idx[idx_n]], temp)     
                #bottom = args.delta*torch.ones(E_new.size(0), dtype=E_new.dtype, device=E_new.device)
                E_new[idx[idx_n]] = torch.max((E_new[idx[idx_n]]+temp)/2, bottom[idx[idx_n]])# use mean to refine the margin to reduce the effect of augmentation on margins
        #-----------------------------------------------------------------------
        expand=(flag1==1)&(flag2==1)
        # smooth strategy 2
        #if expand.sum().item() > sample_count_train*updateRate:
        #    print ("updating the margins...........................................")
        #    IMA_update_margin(E, delta, noise, flag1, flag2, E_new) 
        loss_train = sum(loss_list) / dataset.__len__()
        loss_train_list.append(loss_train)
        logger.info("Epoch {} Training loss {}".format(epoch, loss_train))
        
        
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
            
            fig,axs = plt.subplots(1,4, figsize=(20,5))

            #ax = fig.add_subplot(111)
            X = list(range(epoch+1))
            axs[0].plot(X, loss_train_list, color=cols[0], label="Training Loss")
            axs[1].plot(X, loss_val_list, color=cols[1], label="Validation Loss")
            axs[2].plot(X, MRE_list, color=cols[2], label="MRE")
            axs[3].hist(E.cpu().numpy(), bins=100, range=(0, noise))
            axs[0].set_xlabel("epoch")   
            axs[1].set_xlabel("epoch")
            axs[2].set_xlabel("epoch")
            axs[3].set_xlabel("margins")
            axs[0].legend()
            axs[1].legend()
            axs[2].legend()           
            fig.savefig(runs_dir +"/training.png")
            #save the last epoch
            config['last_epoch'] = epoch

        # dump yaml
        with open(runs_dir + "/config.yaml", "w") as f:
            yaml.dump(config, f)

    # # Test
    # tester.test(net)
