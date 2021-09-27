import argparse
import csv
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import BCELoss
import os
import yaml
import yamlloader
import random

from network import UNet, UNet_Pretrained
from data_loader import Cephalometric
from mylogger import get_mylogger, set_logger_dir
from eval import Evaluater
from utils import to_Image, voting, visualize, make_dir
from attack import FGSMAttack
from statsmodels.robust.scale import huber


def L1Loss(pred, gt, mask=None):
    assert(pred.shape == gt.shape)
    gap = pred - gt
    distence = gap.abs()
    if mask is not None:
        distence = distence * mask
    return distence.sum() / mask.sum()
    # return distence.mean()
def myL1Loss(pred, gt, mask=None,reduction = "mean"):
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
        return distence.sum([1,2,3])/mask.sum([1,2,3])



class Tester(object):
    def __init__(self, logger, config, net=None, tag=None, train="", args=None):
        mode = "Test1" if train == "" else "Train"
        self.datapath = config['dataset_pth']
        
        self.nWorkers = config['num_workers']
    
        self.model = net 


    def getThresholds(self, net=None):
        #self.evaluater.reset()
        if net is not None:
            self.model = net
        assert(hasattr(self, 'model'))
        ID = 0

        distance_list = dict()
        mean_list = dict()
        for i in range(19):
            distance_list[i] = list()
            mean_list[i] = list()

        loss_logic_fn = BCELoss()
        loss_regression_fn = L1Loss
        
        r_list = list()
        ry_list = list()   
        rx_list = list()
        bce_list = list()
        
        dataset_train = Cephalometric(self.datapath, "train")
        dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=False, num_workers=self.nWorkers)
        
        for img, mask, guassian_mask, offset_y, offset_x, landmark_list in tqdm(dataloader_train):
            img, mask, offset_y, offset_x, guassian_mask = img.cuda(), mask.cuda(), \
                offset_y.cuda(), offset_x.cuda(), guassian_mask.cuda()
            with torch.no_grad():
                heatmap, regression_y, regression_x = self.model(img)
                # get the threshold for each of the samples
                loss_regression_fn = myL1Loss
                # the loss for heatmap
                #BCE loss
                loss_logic_fn = BCELoss()
                bce= loss_logic_fn(heatmap, guassian_mask)
                # r rario
                #logic_loss = loss_logic_fn(heatmap, guassian_mask, mask, reduction = "none")
                guassian_mask=guassian_mask/torch.norm(guassian_mask, p=2, keepdim=True)
                heatmap=heatmap/torch.norm(heatmap, p=2, keepdim=True)
                r=(heatmap*guassian_mask).sum(dim=(2,3))
                r=r.mean(dim = 1)
                # the loss for offset
                regression_loss_ys = loss_regression_fn(regression_y, offset_y, mask, reduction = "none")
                regression_loss_xs = loss_regression_fn(regression_x, offset_x, mask, reduction = "none")
               

                # acc them
                r_list.append(r.cpu().numpy())
                ry_list.append(regression_loss_ys.cpu().numpy())
                rx_list.append(regression_loss_xs.cpu().numpy())
                bce_list.append(bce.cpu().item())

        rList = np.concatenate(r_list)
        ryList = np.concatenate(ry_list)
        rxList = np.concatenate(rx_list)
        bcelist = bce_list
        import matplotlib.pyplot as plt
        cols = ['b','g','r','y','k','m','c']
        fig, ax = plt.subplots(1,4, figsize=(20,5))
        ax[0].hist(rList,  bins=50, color=cols[0], label="distribution of ratio")
        ax[1].hist(ryList,bins=50, color=cols[1], label="distribution of offset y error")
        ax[2].hist(rxList,bins=50, color=cols[2], label="distribution of offset x error")
        ax[3].hist(bcelist,bins=50, color=cols[3], label="distribution of BCE")
        ax[0].legend()
        ax[1].legend()
        ax[2].legend() 
        ax[3].legend()
        fig.savefig("./threshold_distribution_of_training_data.png")
        
        return estimateMean(rList), estimateMean(ryList), estimateMean(rxList)       

def estimateMean(l):
    mean,std = np.mean(l),np.std(l)
    # one z score
    return huber(l)[0].item()+3*huber(l)[1].item()
if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    # Parse command line options
    parser = argparse.ArgumentParser(description="get the threshold from already trained base model")
    parser.add_argument("--tag", default='getThreshold', help="position of the output dir")
    parser.add_argument("--debug", default='', help="position of the output dir")
    parser.add_argument("--iteration", default='', help="position of the output dir")
    parser.add_argument("--attack", default='', help="position of the output dir")
    parser.add_argument("--config_file", default="config.yaml", help="default configs")
    parser.add_argument("--checkpoint_file", default="", help="default configs")
    parser.add_argument("--output_file", default="", help="default configs")
    parser.add_argument("--train", default="", help="default configs")
    parser.add_argument("--rand", default="", help="default configs")
    parser.add_argument("--epsilon", default="8", help="default configs")
    args = parser.parse_args()

    with open(os.path.join("./results/base_400_320", args.config_file), "r") as f:
        config = yaml.load(f, Loader=yamlloader.ordereddict.CLoader)
    
    # Create Logger
    logger = get_mylogger()
        

    iteration = 149
    
    # Load model
    # net = UNet(3, config['num_landmarks']).cuda()
    # net = Runnan(3, config['num_landmarks']).cuda()
    net = UNet_Pretrained(3, config['num_landmarks']).cuda()

    logger.info("Loading checkpoints from epoch {}".format(iteration))
    checkpoints = torch.load("./results/base_400_320/model_epoch_{}.pth".format(iteration))
    newCP = dict()
    #adjust the keys(remove the "module.")
    for k in checkpoints.keys():
        newK = ""
        if "module." in k:
            newK = ".".join(k.split(".")[1:])
        else:
            newK = k
        newCP[newK] = checkpoints[k]
    
    #
    net.load_state_dict(newCP)
    net = torch.nn.DataParallel(net)
    #tester = Tester(logger, config, net, args.tag, args.train, args)
    tester = Tester(logger, config, tag=args.tag)
    t1, t2, t3 = tester.getThresholds(net)
    logger.info("the threshold 1 is {}, threshold 2 ry is {}, threshold 3 rx is {}".format(t1, t2, t3))
    # go through all the training set to get the thresholds, three
    
    

