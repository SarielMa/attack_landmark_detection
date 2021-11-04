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
from torch import optim
from os.path import exists

def cal_dice(Mp, M, reduction='none'):
    #Mp.shape  NxKx128x128
    intersection = (Mp*M).sum(dim=(2,3))
    dice = (2*intersection) / (Mp.sum(dim=(2,3)) + M.sum(dim=(2,3)))
    if reduction == 'mean':
        dice = dice.mean()
    elif reduction == 'sum':
        dice = dice.sum()
    return dice

def dice_loss(Mp, M, reduction='none'):
    score=cal_dice(Mp, M, reduction)
    return 1-score

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
#%%
def normalize_noise_(noise, norm_type, eps=1e-8):
    if noise.size(0) == 0:
        return noise
    with torch.no_grad():
        N=noise.view(noise.size(0), -1)
        if norm_type == np.inf or norm_type == 'Linf':
            linf_norm=N.abs().max(dim=1, keepdim=True)[0]
            N *= 1/(linf_norm+eps)
        elif norm_type == 2 or norm_type == 'L2':
            l2_norm=torch.sqrt(torch.sum(N**2, dim=1, keepdim=True))
            l2_norm = torch.max(l2_norm, torch.tensor(eps, dtype=l2_norm.dtype, device=l2_norm.device))
            N *= 1/l2_norm
        else:
            raise NotImplementedError("not implemented.")
    return noise
#%%
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
        return distence.sum([1,2,3])/mask.sum([1,2,3])
    
def heatmap_dice_loss(heatmap, guassian_mask, regression_y, offset_y, regression_x, offset_x, mask):        
    logic_loss = dice_loss(heatmap, mask, reduction='mean')
    return  logic_loss

    
def reg_loss(heatmap, guassian_mask, regression_y, offset_y, regression_x, offset_x, mask):

    loss_regression_fn = L1Loss
    regression_loss_y = loss_regression_fn(regression_y, offset_y, mask, reduction = "mean")
    regression_loss_x = loss_regression_fn(regression_x, offset_x, mask, reduction = "mean")
    return  regression_loss_x + regression_loss_y
    

def total_loss(heatmap, guassian_mask, regression_y, offset_y, regression_x, offset_x, mask):       
    loss_regression_fn = L1Loss
    logic_loss = dice_loss(heatmap, mask, reduction='mean')
    regression_loss_y = loss_regression_fn(regression_y, offset_y, mask, reduction = "mean")
    regression_loss_x = loss_regression_fn(regression_x, offset_x, mask, reduction = "mean")
    return  logic_loss*0.5 + regression_loss_x + regression_loss_y




#%%
def pgd_attack(net, img, mask, offset_y, offset_x, guassian_mask, noise_norm, norm_type, max_iter, step,
               rand_init=True, rand_init_norm=None, targeted=False,
               clip_X_min=-1, clip_X_max=1, use_optimizer=False, loss_fn=None):
    #-----------------------------------------------------
    if loss_fn is None :
        raise ValueError('loss_fn is unkown')
    #-----------------
    img = img.detach()
    #-----------------
    if rand_init == True:
        init_norm=rand_init_norm
        if rand_init_norm is None:
            init_norm=noise_norm
        noise_init=get_noise_init(norm_type, noise_norm, init_norm, img)
        Xn = img + noise_init
    else:
        Xn = img.clone().detach() # must clone
    #-----------------
    noise_new=(Xn-img).detach()
    if use_optimizer == True:
        optimizer = optim.Adamax([noise_new], lr=step)
    #-----------------
    for n in range(0, max_iter):
        Xn = Xn.detach()
        Xn.requires_grad = True
        heatmap, regression_y, regression_x = net(Xn)
        loss= loss_fn(heatmap, guassian_mask, regression_y, offset_y, regression_x, offset_x, mask)
        #---------------------------
        if targeted == True:
            loss=-loss
        #---------------------------
        #loss.backward() will update W.grad
        grad_n=torch.autograd.grad(loss, Xn)[0]
        grad_n=normalize_grad_(grad_n, norm_type)
        if use_optimizer == True:
            noise_new.grad=-grad_n.detach() #grad ascent to maximize loss
            optimizer.step()
        else:
            Xnew = Xn.detach() + step*grad_n.detach()
            noise_new = Xnew-img
        #---------------------
        clip_norm_(noise_new, norm_type, noise_norm)
        Xn = torch.clamp(img+noise_new, clip_X_min, clip_X_max)
        #Xn = img + noise_new
        #noise_new.data -= noise_new.data-(Xn-img).data
        Xn=Xn.detach()
    #---------------------------
    return Xn

class Tester(object):
    def __init__(self,logger, config, testset):
        self.datapath = config['dataset_pth']        
        self.nWorkers = config['num_workers']    
        self.logger = logger
        self.dataset_val = Cephalometric(self.datapath, testset)
        self.dataloader_val = DataLoader(self.dataset_val, batch_size=1, shuffle=False, num_workers=self.nWorkers)


    def validate(self, net, loss_fn, noise=0, norm_type = 2, max_iter = 100):
     

        loss_logic_fn = BCELoss()
        loss_regression_fn = L1Loss
        
        train_loss_list = list()
        regression_loss_list = list()   
        logic_loss_list = list()

        dataloader_val = self.dataloader_val
        dataset_val = self.dataset_val
        evaluater = Evaluater(self.logger, dataset_val.size, dataset_val.original_size)
        Radius = dataset_val.Radius
        #net.eval()
        for img, mask, guassian_mask, offset_y, offset_x, landmark_list in tqdm(dataloader_val):
            img, mask, offset_y, offset_x, guassian_mask = img.cuda(), mask.cuda(), \
                offset_y.cuda(), offset_x.cuda(), guassian_mask.cuda()
            
            if noise > 0:
                step = 5*noise/max_iter
                img = pgd_attack(net, img, mask, offset_y, offset_x, guassian_mask, noise, norm_type, max_iter, step, \
                                  loss_fn = loss_fn)
            with torch.no_grad():    
                heatmap, regression_y, regression_x = net(img)               
                pred_landmark = voting(heatmap, regression_y, regression_x, Radius)   
                evaluater.record(pred_landmark, landmark_list)    
        MRE, _ = evaluater.my_cal_metrics()

        return MRE


if __name__ == "__main__":
    random.seed(10)

    # Parse command line options
    parser = argparse.ArgumentParser(description="get the threshold from already trained base model")
    parser.add_argument("--tag", default='1101_Test1', help="position of the output dir")
    parser.add_argument("--debug", default='', help="position of the output dir")
    parser.add_argument("--iteration", default='', help="position of the output dir")
    parser.add_argument("--attack", default='', help="position of the output dir")
    parser.add_argument("--config_file", default="config.yaml", help="default configs")
    parser.add_argument("--checkpoint_file", default="", help="default configs")
    parser.add_argument("--output_file", default="", help="default configs")
    parser.add_argument("--train", default="", help="default configs")
    parser.add_argument("--rand", default="", help="default configs")
    parser.add_argument("--epsilon", default="8", help="default configs")
    parser.add_argument("--testset", default="Test1")
    parser.add_argument("--cuda", default="0")
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=args.cuda
    iteration = 499
    
    
    #file folders================
    #folders = ["IMA_5_DSH2302Zd50","IMA_5_DSH2302Zd20","IMA_5_DSH2302Zd10","PGD_1_500", "PGD_3_500","PGD_5_500"]
    folders = ["PGD_2_500","IMA_5_DSH500Maxd5","IMA_5_DSH2302Zd50_1000","PGD_1_500", "PGD_3_500"]
    #folders = ["PGD_1_500"]
    #subfolder = "non_pretrain_dice_loose_PGD/"
    subfolder = "non_pretrain_dice_strict_PGD/"
    resultFolder = args.tag
    #========================
    import matplotlib.pyplot as plt
    #plt.figure(figsize = (5,5))
    cm = plt.get_cmap("gist_rainbow")
    noises = [0,0.5,1,1.5,2,2.5,3]
    #attacks=================================================================
    losses = [heatmap_dice_loss, reg_loss, total_loss]
    attackNames = ["DiceOnly", "RegOnly","TotalLoss"]
    # check if folders and pts exist=========================================
    for f in folders:
        print ("exist ",f)
        assert( exists("./results/"+subfolder+f+"/model_epoch_{}.pth".format(iteration)))
    print ("all files exist, test begins...")
    resultDir = os.path.join("./results/"+subfolder,resultFolder)
    assert(not exists(resultDir))
    os.mkdir(resultDir)
    # test starts
    for n, loss_fn in enumerate(losses):
        attackName = attackNames[n]
        rows1 = []
        plt.figure(figsize = (5,5))

        for i, folder in enumerate(folders):
            MRE_list =list()
            
            with open(os.path.join("./results/"+subfolder+folder, args.config_file), "r") as f:
                config = yaml.load(f, Loader=yamlloader.ordereddict.CLoader)     
            # Create Logger
            logger = get_mylogger()
            # Load model
            net = UNet_Pretrained(3, config['num_landmarks']).cuda()   
            print("=======================================================================================")
            print ("the model is {}".format(folder))
            print("Loading checkpoints from epoch {}".format(iteration))
            checkpoints = torch.load("./results/"+subfolder+folder+"/model_epoch_{}.pth".format(iteration))
            newCP = dict()
            #adjust the keys(remove the "module.")
            for k in checkpoints.keys():
                newK = ""
                if "module." in k:
                    newK = ".".join(k.split(".")[1:])
                else:
                    newK = k
                newCP[newK] = checkpoints[k]
            # test
            net.load_state_dict(newCP)
            net = torch.nn.DataParallel(net)
            for noise in noises:
                #net = torch.nn.DataParallel(net)
                #tester = Tester(logger, config, net, args.tag, args.train, args)
                print ("this noise level is {}++++++++++++++++++++++++++++++++++++++++++++++++++++++++++".format(noise))
                tester = Tester(logger, config, testset = args.testset)
                MRE= tester.validate(net, noise = noise, loss_fn = loss_fn)
                print("Testing MRE {}".format(MRE))
                MRE_list.append(MRE)

            plt.yscale("log")                           
            plt.plot(noises,MRE_list, color = cm(1.0*i/len(folders)), label = folder )                                     
            plt.ylabel("log MRE (mm)")
            plt.xlabel("noise (L2)")
            plt.title(attackName)
            plt.legend()
            rows1.append([folder]+[str(round(i,3)) for i in MRE_list])
            
        plt.savefig(os.path.join(resultDir,attackName+"result.pdf"),bbox_inches='tight')

        fields1 = ["noise"]+[str(i) for i in noises]
        with open(os.path.join(resultDir,attackName+"result_MRE.csv"),'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(fields1)
            csvwriter.writerows(rows1) 
        
        
        
        
    
    

