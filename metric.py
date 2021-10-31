import torch
from torch.nn import BCELoss

def cal_dice(Mp, M, reduction='none'):
    #Mp.shape  NxKx128x128
    intersection = (Mp*M).sum(dim=(2,3))
    dice = (2*intersection) / (Mp.sum(dim=(2,3)) + M.sum(dim=(2,3)))
    if reduction == 'mean':
        dice = dice.mean()
    elif reduction == 'sum':
        dice = dice.sum()
    return dice

def dice_loss(Mp, M, reduction='mean'):
    score=cal_dice(Mp, M, reduction)
    return 1-score


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
        logic_loss = logic_loss.view(logic_loss.size(0),-1).mean(1)
        # the loss for offset
        regression_loss_y = loss_regression_fn(regression_y, offset_y, mask, reduction = "none")
        regression_loss_x = loss_regression_fn(regression_x, offset_x, mask, reduction = "none")
        return  regression_loss_x + regression_loss_y + logic_loss * lamb, regression_loss_x + regression_loss_y
    
def total_loss_dice(heatmap, guassian_mask, regression_y, offset_y, regression_x, offset_x, mask, lamb=0.5, reduction = 'sum'):
    # loss
    if reduction == 'sum':

        loss_regression_fn = L1Loss
        # the loss for heatmap
        logic_loss = dice_loss(heatmap, mask)
        # the loss for offset
        regression_loss_y = loss_regression_fn(regression_y, offset_y, mask, reduction = "mean")
        regression_loss_x = loss_regression_fn(regression_x, offset_x, mask, reduction = "mean")
        return  regression_loss_x + regression_loss_y + logic_loss * lamb, regression_loss_x + regression_loss_y
    else: 
        # every sample has its loss, none reduction

        loss_regression_fn = L1Loss
        # the loss for heatmap
        logic_loss = dice_loss(heatmap, mask, reduction = reduction)
        logic_loss = logic_loss.view(logic_loss.size(0),-1).mean(1)
        # the loss for offset
        regression_loss_y = loss_regression_fn(regression_y, offset_y, mask, reduction = "none")
        regression_loss_x = loss_regression_fn(regression_x, offset_x, mask, reduction = "none")
        return  regression_loss_x + regression_loss_y + logic_loss * lamb, regression_loss_x + regression_loss_y

    
def l1_matric(heatmap, guassian_mask, regression_y, offset_y, regression_x, offset_x, mask):
    #loss_logic_fn = L1Loss
    loss_regression_fn = L1Loss

    with torch.no_grad():
        guassian_mask=guassian_mask/torch.norm(guassian_mask, p=2,dim = (2,3), keepdim=True)
        heatmap=heatmap/torch.norm(heatmap, p=2,dim = (2,3), keepdim=True)
        r=(heatmap*guassian_mask).sum(dim=(2,3))
        r=r.mean(dim = 1)
        # the loss for offset
        regression_loss_ys = loss_regression_fn(regression_y, offset_y, mask, reduction = "none")
        regression_loss_xs = loss_regression_fn(regression_x, offset_x, mask, reduction = "none")
    return  r, regression_loss_ys,regression_loss_xs

def l1_matric_v2(heatmap, guassian_mask, regression_y, offset_y, regression_x, offset_x, mask):
    #loss_logic_fn = L1Loss
    loss_regression_fn = L1Loss
    loss_logic_fn = BCELoss(reduction = 'none')
    with torch.no_grad():    
        bce= loss_logic_fn(heatmap, guassian_mask)
        bce = bce.view(bce.size(0),-1).mean(1)
        # the loss for offset
        regression_loss_ys = loss_regression_fn(regression_y, offset_y, mask, reduction = "none")
        regression_loss_xs = loss_regression_fn(regression_x, offset_x, mask, reduction = "none")
    return  bce, regression_loss_ys,regression_loss_xs

def l1_matric_dice(heatmap, guassian_mask, regression_y, offset_y, regression_x, offset_x, mask):
    #loss_logic_fn = L1Loss
    loss_regression_fn = L1Loss

    with torch.no_grad():    
        de= dice_loss(heatmap, mask,reduction = 'none')
        de = de.view(de.size(0),-1).mean(1)
        # the loss for offset
        regression_loss_ys = loss_regression_fn(regression_y, offset_y, mask, reduction = "none")
        regression_loss_xs = loss_regression_fn(regression_x, offset_x, mask, reduction = "none")
    return  de, regression_loss_ys,regression_loss_xs

