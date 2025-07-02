import copy
import torch
import numpy as np
from torch_radon import RadonFanbeam
import torchvision.transforms.v2 as v2
import torch.nn.functional as F



# pre-define the scanning fan-beam geometry
def fanbeam_gen(sparsity, img_size, bias=0, det_spacing=1.2858, pixel_spacing=1.4285, source_det_distance=1085.6, source_distance=595.0, det_count=None):
    if det_count == None: det_count = img_size
    
    (sparse_scan, full_scan) = sparsity
    index = full_scan // sparse_scan
    assert full_scan % sparse_scan == 0

    angles = np.linspace(0, 2 * np.pi, full_scan, endpoint=False)
    seq = np.arange(0, full_scan, 1)
    if bias < 0: bias = len(seq) + bias
    seq = np.roll(seq, -bias) 
    result_seq = seq[::index]
    angles = angles[result_seq]  # roll sinogram for data augmentation (sometimes is equivalent to image rotation)
    
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
        

# data augmentation
transforms = v2.Compose([v2.RandomHorizontalFlip(p=0.5), v2.RandomVerticalFlip(p=0.5), v2.RandomRotation(degrees=180, 
                        interpolation=v2.InterpolationMode.BILINEAR)])

def pre_process(batch, image_size=256, val_flag=False):
    img = batch["image"].cuda()
    IMG = []
    if not val_flag: 
        for i in range(0,len(img)):
            IMG.append(transforms(img[i].unsqueeze(0)))
        img = F.adaptive_avg_pool2d(torch.cat(IMG, dim=0), (image_size, image_size))
    else:
        img = F.adaptive_avg_pool2d(img, (image_size, image_size))
    return img












# CG descent
def WLS_stepSize_ridge(grad, d, Pd, lambda_reg, d_norm_sq=None):
    # Numerator: d^T * grad
    num = torch.sum(d * grad, dim=(2,3), keepdim=True)  # [b,1,1,1]
    
    # Denominator: Pd^T * Pd + 2 * lambda_reg * ||d||^2
    denomA = torch.sum(Pd**2, dim=(2,3), keepdim=True)
    if d_norm_sq is None:
        d_norm_sq = torch.sum(d**2, dim=(2,3), keepdim=True)
    denomB = 2 * lambda_reg * d_norm_sq
    denom = denomA + denomB + 1e-10  # Improve numerical stability
    stepSize = num / denom
    return stepSize


# With optional Tikhonov regularization
def WLS_ridge(sino, ct, numStore, radon, lambda_reg=0):
    b, c, h, w = ct.shape
    device = ct.device
    numIter = max(numStore)
    conjGradRestart = 50

    # Save the original image as a regularization prior
    ct_prior = ct.clone().detach()  # Detach from computation graph

    # Initialize variables
    Pf = radon.forward(ct)
    ct = torch.clamp(ct, min=0)

    # Batch-friendly initial scaling
    Pf_dot_Pf = torch.sum(Pf**2, dim=(2,3), keepdim=True)
    sino_dot_Pf = torch.sum(sino * Pf, dim=(2,3), keepdim=True)

    # Vectorized condition handling
    cond_mask = (Pf_dot_Pf > 1e-8) & (sino_dot_Pf > 1e-8)
    scale_factor = torch.where(cond_mask, sino_dot_Pf / Pf_dot_Pf, torch.zeros_like(Pf_dot_Pf))

    ct = torch.where(cond_mask, ct * scale_factor, ct)
    Pf = torch.where(cond_mask, Pf * scale_factor, torch.zeros_like(Pf))
    Pf_minus_sino = Pf - sino

    # Pre-allocate historical variables
    grad_old_dot_grad_old = torch.zeros(b,1,1,1, device=device)
    Q = 1.0
    grad = torch.zeros_like(ct)
    d = torch.zeros_like(ct)
    grad_old = torch.zeros_like(ct)

    CT = []
    for n in range(numIter+1):
        # Residual projection
        WPf_minus_sino = Pf_minus_sino
        # Compute gradient = data fidelity term + regularization term
        grad_fidelity = radon.backprojection(WPf_minus_sino)
        grad_reg = 2 * lambda_reg * (ct - ct_prior)  # Tikhonov regularization gradient
        grad = grad_fidelity + grad_reg

        # Conjugate gradient update
        u = Q * grad

        # Restart logic
        if n == 0 or (n % conjGradRestart) == 0:
            d = u.clone()
        else:
            # Combined sum operation
            numerator = torch.sum(u * (grad - grad_old), dim=(2,3), keepdim=True)
            gamma = numerator / (grad_old_dot_grad_old + 1e-10)
            d = u + gamma * d

            # Vectorized validity check
            valid_mask = (torch.sum(d * grad, dim=(2,3), keepdim=True) > 0)
            d = torch.where(valid_mask, d, u)

        # Update historical variables
        grad_old_dot_grad_old = torch.sum(u * grad, dim=(2,3), keepdim=True)
        grad_old.copy_(grad)  # In-place copy

        # Compute step size (considering regularization)
        Pd = radon.forward(d)
        d_norm_sq = torch.sum(d**2, dim=(2,3), keepdim=True)  # Pre-compute ||d||^2
        stepSize = WLS_stepSize_ridge(grad, d, Pd, lambda_reg, d_norm_sq)

        # In-place update of ct
        ct.sub_(stepSize * d)
        ct.clamp_(min=0)

        # Update projection
        Pf = radon.forward(ct)
        Pf_minus_sino = Pf - sino

        # Store result
        if n in numStore:
            CT.append(ct.clone())

    return torch.cat(CT, dim=1) if CT else ct


