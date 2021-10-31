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
from PGD import pgd_attack
from metric import total_loss_dice as total_loss


#
def run_model_adv_reg(net, img, mask, offset_y, offset_x, guassian_mask, return_loss=False, reduction='none'):
    heatmap, regression_y, regression_x = net(img)
    
    if return_loss == True:
        loss, _=total_loss(heatmap, guassian_mask, regression_y, offset_y, regression_x, offset_x, mask, reduction = reduction)       
        return heatmap, regression_y, regression_x, loss
    else:
        return heatmap, regression_y, regression_x
    
def classify_model_std_output_reg(heatmap, guassian_mask, regression_y, offset_y, regression_x, offset_x, mask, threshold="none"):
    #useless
    threshold1 = 10000000

    loss, _=total_loss(heatmap, guassian_mask, regression_y, offset_y, regression_x, offset_x, mask,  reduction='none') 
    Yp_e_Y=(loss<=threshold1) 
    #Yp_e_Y=(r>=threshold1) & (ry <=threshold2) & (rx <= threshold3)
    return Yp_e_Y
#%%

if __name__ == "__main__":

    #device = torch.device('cuda:1')
    # Parse command line options
    parser = argparse.ArgumentParser(description="Train Unet landmark detection network")
    parser.add_argument("--tag", default='PGD_10_', help="name of the run")
    parser.add_argument("--cuda", default='0', help="cuda id")
    parser.add_argument("--config_file", default="config.yaml", help="default configs")
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
    
    # net = UNet(3, config['num_landmarks'])
    net = UNet_Pretrained(3, config['num_landmarks'])
    #net = torch.nn.DataParallel(net)
    net = net.cuda()
    logger.info(net)

    optimizer = optim.Adam(params=net.parameters(), \
        lr=config['learning_rate'], betas=(0.9,0.999), eps=1e-08, weight_decay=1e-4)
    
    scheduler = StepLR(optimizer, config['decay_step'], gamma=config['decay_gamma'])


    # Tester
    tester = Tester(logger, config, tag=args.tag)
    
    # parameters
 
    #======================
    noise = float(args.tag.split("_")[1])
    assert(type(noise) == float)
    norm_type = 2
    max_iter = 20
    step = 5*noise/max_iter
    title = "PGD"
    #======================
    
    loss_train_list = list()
    loss_val_list = list()
    MRE_list = list()
    
    for epoch in range(config['num_epochs']):
        loss_list = list()
        regression_loss_list = list()

        net.train()
        for img, mask, guassian_mask, offset_y, offset_x, landmark_list, idx in tqdm(dataloader):
            img, mask, offset_y, offset_x, guassian_mask = img.cuda(), mask.cuda(), \
                offset_y.cuda(), offset_x.cuda(), guassian_mask.cuda()
                
            net.zero_grad()
            heatmap, regression_y, regression_x = net(img)
            lossp, regLossp  = total_loss(heatmap, guassian_mask, regression_y, offset_y, regression_x, offset_x, mask)
            
            imgn,_ = pgd_attack(net, img, mask, offset_y, offset_x, guassian_mask, 
                               noise, norm_type, max_iter, step,
                               rand_init_norm=None, rand_init_Xn=None,
                               targeted=False, clip_X_min=-1, clip_X_max=1,
                               refine_Xn_max_iter=10,
                               Xn1_equal_X=False, Xn2_equal_Xn=False,
                               stop_near_boundary=False,
                               stop_if_label_change=False,
                               stop_if_label_change_next_step=False,
                               use_optimizer=False,
                               run_model=run_model_adv_reg, classify_model_output=classify_model_std_output_reg,               
                               model_eval_attack=False)
            
            heatmapn, regression_yn, regression_xn = net(imgn)
            lossn, regLossn  = total_loss(heatmapn, guassian_mask, regression_yn, offset_y, regression_xn, offset_x, mask)
            
            loss = lossp*0.5 + lossn*0.5
            
            loss.backward()
            optimizer.step()
            loss_list.append(loss)
         #--------------------update the margins

        #-----------------------------------------------------------------------

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
            fig.savefig(runs_dir +"/training.png")
            #save the last epoch
            config['last_epoch'] = epoch

        # dump yaml
        with open(runs_dir + "/config.yaml", "w") as f:
            yaml.dump(config, f)

    # # Test
    # tester.test(net)
