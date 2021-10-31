import torch
from torch import optim
import numpy as np

def IMA_loss(net, img, mask, offset_y, offset_x, guassian_mask, 
             margin, norm_type, max_iter, step,
             rand_init_norm=None, rand_init_Xn=None,
             clip_X_min=-1, clip_X_max=1,
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
        temp=tuple([1]*len(img[0].size()))
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
        loss1 = loss_X[Yp_ne_Y].sum()/Yp_ne_Y.sum().item()
    if Yp_e_Y.sum().item()>0:
        loss2 = loss_X[Yp_e_Y].sum()/Yp_e_Y.sum().item()
    #---------------------------------
    train_mode=net.training# record the mode
    #---------------------------------
    # we ingore the re_initial, there is no need to use enable_loss3
    
    enable_loss3=False
    if Yp_e_Y.sum().item()>0 and beta>0:
         enable_loss3=True
    
    #----------------------------------
    if enable_loss3 == True:
        #print ("loss3 is enabled...with n samples ",Yp_e_Y.sum().item() )
        Xn, advc = repeated_pgd_attack(net, img, mask, offset_y, offset_x, guassian_mask, 
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

def pgd_attack_lower(net, img, mask, offset_y, offset_x, guassian_mask, 
               noise_norm, norm_type, max_iter, step,
               rand_init_norm=None, rand_init_Xn=None,
               targeted=False, clip_X_min=-1, clip_X_max=1,
               refine_Xn_max_iter=10,
               Xn1_equal_X=False, Xn2_equal_Xn=False,
               stop_near_boundary=False,
               stop_if_label_change=False,
               stop_if_label_change_next_step=False,
               use_optimizer=False,
               run_model=None, classify_model_output=None,               
               model_eval_attack=False):
    #-------------------------------------------
    train_mode=net.training# record the mode
    if model_eval_attack == True and train_mode == True:
        net.eval()#set model to evaluation mode
    #-----------------
    img = img.detach()
    #-----------------
    advc=torch.zeros(img.size(0), dtype=torch.int64, device=img.device)
    #-----------------
    if rand_init_norm is None:
        rand_init_norm = noise_norm
        
    noise_init=get_noise_init(norm_type, noise_norm, rand_init_norm, img)    
    Xn = img + noise_init
    #-----------------
    Xn1=img.detach().clone()
    Xn2=img.detach().clone()
    Ypn_old_e_Y=torch.ones(guassian_mask.shape[0], dtype=torch.bool, device=guassian_mask.device)
    Ypn_old_ne_Y=~Ypn_old_e_Y
    #-----------------
    noise=(Xn-img).detach()
    if use_optimizer == True:
        optimizer = optim.Adamax([noise], lr=step)
    #-----------------
    for n in range(0, max_iter+1):
        Xn = Xn.detach()
        Xn.requires_grad = True        
        heatmap, regression_y, regression_x, loss=run_model(net, Xn, mask, offset_y, offset_x, guassian_mask, return_loss=True, reduction='sum')
        Ypn_e_Y=classify_model_output(heatmap, guassian_mask, regression_y, offset_y, regression_x, offset_x, mask)
        Ypn_ne_Y=~Ypn_e_Y
        #---------------------------
        #targeted attack, Y should be filled with targeted output
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
        #temp1=(A&A_old)&(advc<1)
        temp1=(A&A_old)
        Xn1[temp1]=Xn[temp1].data# last right and this right
        #temp2=(B&A_old)&(advc<1)
        temp2=(B&A_old)
        Xn2[temp1]=Xn[temp1].data# last right and this right
        Xn2[temp2]=Xn[temp2].data# last right and this wrong
        
        #advc[B]+=1#
        advc[B] = 1
        advc[A] = 0
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
                noise = Xnew-img
            #---------------------
            clip_norm_(noise, norm_type, noise_norm)
            Xn = torch.clamp(img+noise, clip_X_min, clip_X_max)
            #Xn = img+noise
            #noise.data -= noise.data-(Xn-img).data
            #---------------------
            Ypn_old_e_Y=Ypn_e_Y
            Ypn_old_ne_Y=Ypn_ne_Y
    #---------------------------
    Xn_out = Xn.detach()
    if Xn1_equal_X:
        Xn1=img.detach().clone()
    if Xn2_equal_Xn:
        Xn2=Xn
    if stop_near_boundary == True:
        temp=advc>0
        if temp.sum()>0:
            Xn_out=refine_Xn_onto_boundary(net, Xn1, Xn2, mask, offset_y, offset_x, guassian_mask, refine_Xn_max_iter, run_model, classify_model_output)
    #---------------------------
    if train_mode == True and net.training == False:
        net.train()
    #---------------------------
    return Xn_out, advc

def pgd_attack(net, img, mask, offset_y, offset_x, guassian_mask, 
               noise_norm, norm_type, max_iter, step,
               rand_init_norm=None, rand_init_Xn=None,
               targeted=False, clip_X_min=-1, clip_X_max=1,
               refine_Xn_max_iter=10,
               Xn1_equal_X=False, Xn2_equal_Xn=False,
               stop_near_boundary=False,
               stop_if_label_change=False,
               stop_if_label_change_next_step=False,
               use_optimizer=False,
               run_model=None, classify_model_output=None,               
               model_eval_attack=False):
    #-------------------------------------------
    train_mode=net.training# record the mode
    if model_eval_attack == True and train_mode == True:
        net.eval()#set model to evaluation mode
    #-----------------
    img = img.detach()
    #-----------------
    advc=torch.zeros(img.size(0), dtype=torch.int64, device=img.device)
    #-----------------
    if rand_init_norm is None:
        rand_init_norm = noise_norm
        
    noise_init=get_noise_init(norm_type, noise_norm, rand_init_norm, img)    
    Xn = img + noise_init
    #-----------------
    Xn1=img.detach().clone()
    Xn2=img.detach().clone()
    Ypn_old_e_Y=torch.ones(guassian_mask.shape[0], dtype=torch.bool, device=guassian_mask.device)
    Ypn_old_ne_Y=~Ypn_old_e_Y
    #-----------------
    noise=(Xn-img).detach()
    if use_optimizer == True:
        optimizer = optim.Adamax([noise], lr=step)
    #-----------------
    for n in range(0, max_iter+1):
        Xn = Xn.detach()
        Xn.requires_grad = True        
        heatmap, regression_y, regression_x, loss=run_model(net, Xn, mask, offset_y, offset_x, guassian_mask, return_loss=True, reduction='sum')
        Ypn_e_Y=classify_model_output(heatmap, guassian_mask, regression_y, offset_y, regression_x, offset_x, mask)
        Ypn_ne_Y=~Ypn_e_Y
        #---------------------------
        #targeted attack, Y should be filled with targeted output
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
        #temp1=(A&A_old)
        Xn1[temp1]=Xn[temp1].data# last right and this right
        temp2=(B&A_old)&(advc<1)
        #temp2=(B&A_old)
        Xn2[temp1]=Xn[temp1].data# last right and this right
        Xn2[temp2]=Xn[temp2].data# last right and this wrong
        
        advc[B]+=1#
        #advc[B] = 1
        #advc[A] = 0
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
                noise = Xnew-img
            #---------------------
            clip_norm_(noise, norm_type, noise_norm)
            Xn = torch.clamp(img+noise, clip_X_min, clip_X_max)
            #Xn = img+noise
            #noise.data -= noise.data-(Xn-img).data
            #---------------------
            Ypn_old_e_Y=Ypn_e_Y
            Ypn_old_ne_Y=Ypn_ne_Y
    #---------------------------
    Xn_out = Xn.detach()
    if Xn1_equal_X:
        Xn1=img.detach().clone()
    if Xn2_equal_Xn:
        Xn2=Xn
    if stop_near_boundary == True:
        temp=advc>0
        if temp.sum()>0:
            Xn_out=refine_Xn_onto_boundary(net, Xn1, Xn2, mask, offset_y, offset_x, guassian_mask, refine_Xn_max_iter, run_model, classify_model_output)
    #---------------------------
    if train_mode == True and net.training == False:
        net.train()
    #---------------------------
    return Xn_out, advc

#%%
def refine_onto_boundary(net, Xn1, Xn2, mask, offset_y, offset_x, guassian_mask, max_iter, run_model, classify_model_output):
#note: Xn1 and Xn2 will be modified
    with torch.no_grad():
        Xn=(Xn1+Xn2)/2
        for k in range(0, max_iter):
            heatmap, regression_y, regression_x =run_model(net, Xn, mask, offset_y, offset_x, guassian_mask, return_loss=False)            
            Ypn_e_Y=classify_model_output(heatmap, guassian_mask, regression_y, offset_y, regression_x, offset_x, mask)
            Ypn_ne_Y=~Ypn_e_Y
            Xn1[Ypn_e_Y]=Xn[Ypn_e_Y]
            Xn2[Ypn_ne_Y]=Xn[Ypn_ne_Y]
            Xn=(Xn1+Xn2)/2
    return Xn, Xn1, Xn2
#%%
def refine_Xn_onto_boundary(net, Xn1, Xn2, mask, offset_y, offset_x, guassian_mask, max_iter, run_model, classify_model_output):
#note: Xn1 and Xn2 will be modified
    Xn, Xn1, Xn2=refine_onto_boundary(net, Xn1, Xn2, mask, offset_y, offset_x, guassian_mask, max_iter, run_model, classify_model_output)
    return Xn
#%%
def refine_Xn1_onto_boundary(model, Xn1, Xn2, Y, max_iter, run_model, classify_model_output):
#note: Xn1 and Xn2 will be modified
    Xn, Xn1, Xn2=refine_onto_boundary(model, Xn1, Xn2, Y, max_iter, run_model, classify_model_output)
    return Xn1
#%%
def refine_Xn2_onto_boundary(model, Xn1, Xn2, Y, max_iter, run_model, classify_model_output):
#note: Xn1 and Xn2 will be modified
    Xn, Xn1, Xn2=refine_onto_boundary(model, Xn1, Xn2, Y, max_iter, run_model, classify_model_output)
    return Xn2
#%%

def repeated_pgd_attack(net, img, mask, offset_y, offset_x, guassian_mask, 
                        noise_norm, norm_type, max_iter, step,
                        rand_init_norm=None, rand_init_Xn=None,
                        targeted=False, clip_X_min=-1, clip_X_max=1,
                        refine_Xn_max_iter=10,
                        Xn1_equal_X=False,
                        Xn2_equal_Xn=False,
                        stop_near_boundary=False,
                        stop_if_label_change=False,
                        stop_if_label_change_next_step=False,
                        use_optimizer=False,
                        run_model=None, classify_model_output=None,
                        model_eval_attack=False,
                        num_repeats=1):
    for m in range(0, num_repeats):
        Xm, advcm = pgd_attack(net, img, mask, offset_y, offset_x, guassian_mask,
                               noise_norm, norm_type, max_iter, step,
                               rand_init_norm, rand_init_Xn,
                               targeted, clip_X_min, clip_X_max,
                               refine_Xn_max_iter,
                               Xn1_equal_X,
                               Xn2_equal_Xn,
                               stop_near_boundary,
                               stop_if_label_change,
                               stop_if_label_change_next_step,
                               use_optimizer,
                               run_model, classify_model_output,
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

