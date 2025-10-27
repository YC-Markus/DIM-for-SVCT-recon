import copy
import torch
import random
import numpy as np
from torch_radon import Radon, RadonFanbeam
import torchvision.transforms.v2 as v2
import torch.nn.functional as F




def fanbeam_gen(sparsity, img_size, bias=0, det_spacing=1.2858, pixel_spacing=1.4285, det_count=None):
    source_det_distance=1085.6
    source_distance=595.0
    if det_count == None: det_count = img_size
     # 256
    
    
    (sparse_scan, full_scan) = sparsity
    index = full_scan // sparse_scan
    assert full_scan % sparse_scan == 0
    angles = np.linspace(0, 2 * np.pi, full_scan, endpoint=False)
    seq = np.arange(0, full_scan, 1)
    # seq_length = len(seq) // 2 # producing half-scan angle sequence
    if bias < 0: bias = len(seq) + bias
    seq = np.roll(seq, -bias) # [:seq_length]
    result_seq = seq[::index]
    angles = angles[result_seq]
    
    ops_example = RadonFanbeam(resolution=img_size, angles=angles, source_distance=source_distance*pixel_spacing,
                               det_distance=(source_det_distance-source_distance) * pixel_spacing,
                               det_count=det_count, det_spacing=det_spacing * pixel_spacing, clip_to_circle=True)
    return ops_example


def recon(projections, radon): 
    sino = projections
    filtered_sinogram = radon.filter_sinogram(sino, filter_name='ramp')
    fbp = radon.backprojection(filtered_sinogram)
    return fbp


class ModelEMA:
    def __init__(self, model, decay=0.9999):
        self.ema_model = copy.deepcopy(model).eval()  # create a shadow model
        self.decay = decay
    def update(self, model):
        with torch.no_grad():
            for ema_params, model_params in zip(self.ema_model.parameters(), model.parameters()):
                ema_params.data *= self.decay
                ema_params.data += (1.0 - self.decay) * model_params.data
    def get(self):
        self.ema_model.eval()
        return self.ema_model
   

detector_num = 256
transforms = v2.Compose([v2.RandomHorizontalFlip(p=0.5), v2.RandomVerticalFlip(p=0.5), v2.RandomRotation(degrees=180, 
                        interpolation=v2.InterpolationMode.BILINEAR)])
def pre_process(batch, val_flag=False):
    img = batch["image"].cuda()
    IMG = []
    if not val_flag: 
        for i in range(0,len(img)):
            IMG.append(transforms(img[i].unsqueeze(0)))
        img = F.adaptive_avg_pool2d(torch.cat(IMG, dim=0), (detector_num, detector_num))
    else:
        img = F.adaptive_avg_pool2d(img, (detector_num, detector_num))
    return img


def WLS_fast_stepSize(grad, d, Pd, num=None):
    if num is None:
        num = torch.sum(d * grad, dim=(2,3), keepdim=True)  # [b,1,1,1]
    
    denomA = torch.sum(Pd**2, dim=(2,3), keepdim=True)
    denom = denomA + 1e-10
    stepSize = num / denom
    return stepSize


def WLS_fast(sino, ct, numStore, radon):
    b, c, h, w = ct.shape
    device = ct.device
    numIter = max(numStore)
    conjGradRestart = 50

    Pf = radon.forward(ct)
    ct = torch.clamp(ct, min=0)
    
    Pf_dot_Pf = torch.sum(Pf**2, dim=(2,3), keepdim=True)
    sino_dot_Pf = torch.sum(sino * Pf, dim=(2,3), keepdim=True)
    
    cond_mask = (Pf_dot_Pf > 1e-8) & (sino_dot_Pf > 1e-8)
    scale_factor = torch.where(cond_mask, sino_dot_Pf / Pf_dot_Pf, torch.zeros_like(Pf_dot_Pf))
    
    ct = torch.where(cond_mask, ct * scale_factor, ct)
    Pf = torch.where(cond_mask, Pf * scale_factor, torch.zeros_like(Pf))
    Pf_minus_sino = Pf - sino

    grad_old_dot_grad_old = torch.zeros(b,1,1,1, device=device)
    Q = 1.0
    grad = torch.zeros_like(ct)
    d = torch.zeros_like(ct)
    grad_old = torch.zeros_like(ct)
    
    CT = []
    for n in range(numIter+1):
        WPf_minus_sino = Pf_minus_sino
        grad = radon.backprojection(WPf_minus_sino)
        u = Q * grad
        
        if n == 0 or (n % conjGradRestart) == 0:
            d = u.clone()
        else:
            numerator = torch.sum(u * (grad - grad_old), dim=(2,3), keepdim=True)
            gamma = numerator / (grad_old_dot_grad_old + 1e-10)
            d = u + gamma * d
            
            valid_mask = (torch.sum(d * grad, dim=(2,3), keepdim=True) > 0)
            d = torch.where(valid_mask, d, u)

        grad_old_dot_grad_old = torch.sum(u * grad, dim=(2,3), keepdim=True)
        grad_old.copy_(grad)
        
        Pd = radon.forward(d)
        num = torch.sum(d * grad, dim=(2,3), keepdim=True)
        stepSize = WLS_fast_stepSize(grad, d, Pd, num)
        
        ct.sub_(stepSize * d)
        ct.clamp_(min=0)
        
        Pf = radon.forward(ct)
        Pf_minus_sino = Pf - sino

        if n in numStore:
            CT.append(ct.clone())

    return torch.cat(CT, dim=1) if CT else ct



def WLS_fast_stepSize_ridge(grad, d, Pd, lambda_reg, d_norm_sq=None):
    num = torch.sum(d * grad, dim=(2,3), keepdim=True)  # [b,1,1,1]
    
    denomA = torch.sum(Pd**2, dim=(2,3), keepdim=True)
    if d_norm_sq is None:
        d_norm_sq = torch.sum(d**2, dim=(2,3), keepdim=True)
    denomB = 2 * lambda_reg * d_norm_sq
    denom = denomA + denomB + 1e-10
    stepSize = num / denom
    return stepSize


def WLS_fast_ridge(sino, ct, numStore, radon, lambda_reg=0.1):
    b, c, h, w = ct.shape
    device = ct.device
    numIter = max(numStore)
    conjGradRestart = 50

    ct_prior = ct.clone().detach()
    
    Pf = radon.forward(ct)
    ct = torch.clamp(ct, min=0)
    
    Pf_dot_Pf = torch.sum(Pf**2, dim=(2,3), keepdim=True)
    sino_dot_Pf = torch.sum(sino * Pf, dim=(2,3), keepdim=True)
    
    cond_mask = (Pf_dot_Pf > 1e-8) & (sino_dot_Pf > 1e-8)
    scale_factor = torch.where(cond_mask, sino_dot_Pf / Pf_dot_Pf, torch.zeros_like(Pf_dot_Pf))
    
    ct = torch.where(cond_mask, ct * scale_factor, ct)
    Pf = torch.where(cond_mask, Pf * scale_factor, torch.zeros_like(Pf))
    Pf_minus_sino = Pf - sino

    grad_old_dot_grad_old = torch.zeros(b,1,1,1, device=device)
    Q = 1.0
    grad = torch.zeros_like(ct)
    d = torch.zeros_like(ct)
    grad_old = torch.zeros_like(ct)
    
    CT = []
    for n in range(numIter+1):
        WPf_minus_sino = Pf_minus_sino
        grad_fidelity = radon.backprojection(WPf_minus_sino)
        grad_reg = 2 * lambda_reg * (ct - ct_prior) 
        grad = grad_fidelity + grad_reg

        u = Q * grad
        
        if n == 0 or (n % conjGradRestart) == 0:
            d = u.clone()
        else:
            numerator = torch.sum(u * (grad - grad_old), dim=(2,3), keepdim=True)
            gamma = numerator / (grad_old_dot_grad_old + 1e-10)
            d = u + gamma * d
            
            valid_mask = (torch.sum(d * grad, dim=(2,3), keepdim=True) > 0)
            d = torch.where(valid_mask, d, u)

        grad_old_dot_grad_old = torch.sum(u * grad, dim=(2,3), keepdim=True)
        grad_old.copy_(grad) 
        
        Pd = radon.forward(d)
        d_norm_sq = torch.sum(d**2, dim=(2,3), keepdim=True)
        stepSize = WLS_fast_stepSize_ridge(grad, d, Pd, lambda_reg, d_norm_sq)
        
        ct.sub_(stepSize * d)
        ct.clamp_(min=0)
        
        Pf = radon.forward(ct)
        Pf_minus_sino = Pf - sino

        if n in numStore:
            CT.append(ct.clone())

    return torch.cat(CT, dim=1) if CT else ct




def evaluate(ct_pred, ct_ref, ssim, psnr, lpips, EVAL, resolution):
    ct_pred, ct_ref = torch.clip(ct_pred, min=0, max=1), torch.clip(ct_ref, min=0, max=1)
    EVAL[f'SSIM_{resolution}'].append(ssim(ct_pred, ct_ref).mean().item())
    EVAL[f'PSNR_{resolution}'].append(psnr(ct_pred, ct_ref).mean().item())
    EVAL[f'LPIPS_{resolution}'].append(lpips(ct_pred, ct_ref).mean().item())
    EVAL[f'L1_{resolution}'].append(F.l1_loss(ct_pred, ct_ref, reduction='none').mean().item())
    EVAL[f'L2_{resolution}'].append(F.mse_loss(ct_pred, ct_ref, reduction='none').mean().item())
    return EVAL
