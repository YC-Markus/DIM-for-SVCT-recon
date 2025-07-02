import os
import math
import torch
import logging
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from monai.utils import first
from monai.data import DataLoader
from monai.metrics import PSNRMetric
from generative.metrics import SSIMMetric # monai-generative
from generative.losses import PerceptualLoss
from generative.networks.nets import DiffusionModelUNet
from dataset import *
from utils import *
os.environ['CUDA_VISIBLE_DEVICES'] = "7"
torch.backends.cudnn.benchmark = True


batch_size = 4
num_workers = 32
learning_rate = 2e-4
ema_start = 5 # training epoch that begins using EMA
ema_rate = 0.9995
epoch_end = 90
image_size = 256

log_path = '/workspace/PROJECT/SVCT/DIM/test_log/{}.log'
model_save_path = '/workspace/PROJECT/SVCT/DIM/models/32/{}.pth'

# model_path = '/workspace/PROJECT/SVCT/DIM/models/32/89-ct-32resol.pth'
# state_dict = torch.load(model_path)
# MODEL['CT_32'].load_state_dict(state_dict['model'])


psnr = PSNRMetric(max_val=1.)
ssim = SSIMMetric(spatial_dims=2, data_range=1.)
lpips = PerceptualLoss(spatial_dims=2, network_type="alex").cuda()

train_loader = DataLoader(CTDataset('/workspace/DATASET/AAPM/train_set', mode='train'), batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True)
val_loader = DataLoader(CTDataset('/workspace/DATASET/AAPM/val_set', mode='val'), batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True)
test_loader = DataLoader(CTDataset('/workspace/DATASET/AAPM/test_set', mode='val'), batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True)

VAL_LOSSES_SINO, VAL_LOSSES_CT = [], []

RADON = {}
RADON['32x256'] = fanbeam_gen(sparsity=(32, image_size), img_size=image_size, bias=0) # sparsity = (sparse-view number, full-view number)
RADON['64x256'] = fanbeam_gen(sparsity=(64, image_size), img_size=image_size, bias=0) # if using random rotation to image, rolling bias for sinogram is 0
RADON['128x256'] = fanbeam_gen(sparsity=(128, image_size), img_size=image_size, bias=0)
RADON['256x256'] = fanbeam_gen(sparsity=(256, image_size), img_size=image_size, bias=0)
# !!! very important !!!
# Each time the image is downsampled by a factor of two, the corresponding scanning geometry also have to be adjusted: pixel_spacing should be divided by 2, and det_spacing should be multiplied by 2.
RADON['32x16'] =  fanbeam_gen(sparsity=(32, 32), img_size=16, bias=0, pixel_spacing=1.4285/16, det_spacing=1.2858*16) 
RADON['64x32'] = fanbeam_gen(sparsity=(64, 64), img_size=32, bias=0, pixel_spacing=1.4285/8, det_spacing=1.2858*8)
RADON['128x64'] = fanbeam_gen(sparsity=(128, 128), img_size=64, bias=0, pixel_spacing=1.4285/4, det_spacing=1.2858*4)

MODEL, EMA = {}, {}

MODEL['CT_32'] = DiffusionModelUNet(spatial_dims=2, num_res_blocks=2,
    in_channels=49,
    out_channels=1,
    num_channels=(128, 128, 256),
    attention_levels=(False, True, True),
    num_head_channels=(0, 128, 256),
).cuda()



class DIM_Modules():
    def __init__(self):
        self.CT, self.SINO, self.SINO_MASK = {}, {}, {}


    def pre_cal(self, ct_full):
        self.b, _, _, _ = ct_full.shape
        self.SINO['32x256'] = RADON['32x256'].forward(ct_full)
        self.SINO['64x256'] = RADON['64x256'].forward(ct_full)
        self.SINO['128x256'] = RADON['128x256'].forward(ct_full)

        self.SINO['32x16'] = torch.cat([F.adaptive_avg_pool2d(self.SINO['32x256'], (32,16)) / 16, F.adaptive_max_pool2d(self.SINO['32x256'], (32,16)) / 16, 1-F.adaptive_max_pool2d(1-self.SINO['32x256'], (32,16)) / 16], dim=1)
        self.SINO['32x32'] = torch.cat([F.adaptive_avg_pool2d(self.SINO['32x256'], (32,32)) / 8, F.adaptive_max_pool2d(self.SINO['32x256'], (32,32)) / 8, 1-F.adaptive_max_pool2d(1-self.SINO['32x256'], (32,32)) / 8], dim=1)
        self.SINO['64x32'] = torch.cat([F.adaptive_avg_pool2d(self.SINO['64x256'], (64,32)) / 8, F.adaptive_max_pool2d(self.SINO['64x256'], (64,32)) / 8, 1-F.adaptive_max_pool2d(1-self.SINO['64x256'], (64,32)) / 8], dim=1)
        self.SINO['32x64'] = torch.cat([F.adaptive_avg_pool2d(self.SINO['32x256'], (32,64)) / 4, F.adaptive_max_pool2d(self.SINO['32x256'], (32,64)) / 4, 1-F.adaptive_max_pool2d(1-self.SINO['32x256'], (32,64)) / 4], dim=1)
        self.SINO['64x64'] = torch.cat([F.adaptive_avg_pool2d(self.SINO['64x256'], (64,64)) / 4, F.adaptive_max_pool2d(self.SINO['64x256'], (64,64)) / 4, 1-F.adaptive_max_pool2d(1-self.SINO['64x256'], (64,64)) / 4], dim=1)
        self.SINO['128x64'] = torch.cat([F.adaptive_avg_pool2d(self.SINO['128x256'], (128,64)) / 4, F.adaptive_max_pool2d(self.SINO['128x256'], (128,64)) / 4, 1-F.adaptive_max_pool2d(1-self.SINO['128x256'], (128,64)) / 4], dim=1)

        self.CT['32x32'] = torch.cat([F.adaptive_avg_pool2d(ct_full, (32,32)), F.adaptive_max_pool2d(ct_full, (32,32)), 1-F.adaptive_max_pool2d(1-ct_full, (32,32))], dim=1)



    def predict_sino_32(self, numStore=(2,4,6,9,12,16,20,25,30,36,42,50)):
        self.CT['16x16_fbp'] = recon(self.SINO['32x16'][:,0].unsqueeze(1), RADON['32x16'])
        self.CT['32x32_fbp_interp'] = F.interpolate(self.CT['16x16_fbp'], (32,32), mode='bicubic')
       
        self.CT['32x32_wls_all'] = WLS_ridge(self.SINO['32x32'], torch.clone(self.CT['32x32_fbp_interp']), numStore, RADON['32x32'])
        indices_to_remove = [1,2,7,8,13,14,19,20,25,26,31,32] # Remove some intermediate results from maximal and minimal modalities
        mask = torch.ones(self.CT['32x32_wls_all'].size(1), dtype=torch.bool)  # Initialize all as True
        mask[indices_to_remove] = False  # Mark slices to be deleted as False
        self.CT['32x32_wls'] = self.CT['32x32_wls_all'][:,mask]  # Keep only rows where mask=True
        
        self.CT['32x32_wls_diff'] = self.CT['32x32_wls'] - self.CT['32x32_fbp_interp'].repeat(1,24,1,1)
        return self.CT['32x32_wls'], self.CT['32x32']


    def predict_ct_32(self, ct_model):
        self.timesteps = torch.full((self.CT['32x32_wls'].shape[0],), 0).long().cuda()
        self.CT['32x32_input'] = torch.cat([self.CT['32x32_wls']], self.CT['32x32_wls_diff'], self.CT['32x32_fbp_interp'], dim=1) 
        self.CT['32x32_output'] = ct_model(x = self.CT['32x32_input'], timesteps = self.timesteps) + self.CT['32x32_fbp_interp']
        return self.CT['32x32_output'], self.CT['32x32']


    def loss_cal_ct_32(self, ref):
        loss_CT = F.l1_loss(self.CT['32x32_output'], ref)
        loss_SINO = F.l1_loss(RADON['64x32'].filter_sinogram(RADON['64x32'].forward(self.CT['32x32_output'])), RADON['64x32'].filter_sinogram(RADON['64x32'].forward(ref)))
        return [loss_CT, loss_SINO*10]
    
    
    


def test_ct_32(modules, test_loader, test_model, save_flag=False, epoch='test'):
    with torch.no_grad():
        test_model.eval()
        SSIM, PSNR, LPIPS, L1, L2 = [], [], [], [], []

        for val_step, batch in enumerate(test_loader, start=0):
            ct_full = pre_process(batch, image_size, val_flag=True)
            modules.pre_cal(ct_full)
            modules.predict_sino_32()
            ct_pred_32, ct_ref_32 = modules.predict_ct_32(test_model)
            ct_pred, ct_ref = torch.clip(ct_pred_32, min=0, max=1), torch.clip(ct_ref_32, min=0, max=1)
            
            SSIM.append(ssim(ct_pred, ct_ref).mean().item())
            PSNR.append(psnr(ct_pred, ct_ref).mean().item())
            LPIPS.append(lpips(ct_pred, ct_ref).mean().item())
            L1.append(F.l1_loss(ct_pred, ct_ref).mean().item())
            L2.append(F.mse_loss(ct_pred, ct_ref).mean().item())

        text = f"epoch: {epoch}, ssim: {round(np.mean(SSIM),6)}, psnr: {round(np.mean(PSNR),6)}, lpips: {round(np.mean(LPIPS),6)}, l1: {round(np.mean(L1),6) }, l2: {round(np.mean(L2),6)}"
        print(text)
        logging.info(text)
        if save_flag: 
            VAL_LOSSES_CT.append(np.mean(SSIM))
            if np.mean(SSIM) == np.max(VAL_LOSSES_CT) and ('CT_32' in EMA):
                torch.save({'model': test_model.state_dict()}, model_save_path.format(str(epoch)+'-ct-32resol'))
    test_model.train()





def train_ct_32(modules, train_loader, CT_32):
    optimizer = torch.optim.AdamW(params=CT_32.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 85], gamma=0.25)

    for epoch in range(0, epoch_end):
        CT_32.train()
        epoch_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=120)
        progress_bar.set_description(f"Epoch {epoch}")
        if epoch == ema_start: 
            EMA['CT_32'] = ModelEMA(CT_32, decay=ema_rate)
        for step, batch in progress_bar:
            with torch.no_grad():
                ct_full = pre_process(batch, image_size, val_flag=False)
                modules.pre_cal(ct_full)
                modules.predict_sino_32()

            loss = 0
            optimizer.zero_grad(set_to_none=True)
            _, ref = modules.predict_ct_32(CT_32)
            LOSS = modules.loss_cal_ct_32(ref)
        
            for j in range(len(LOSS)): loss += LOSS[j]
            loss.backward()
            optimizer.step()

            if 'CT_32' in EMA: EMA['CT_32'].update(MODEL['CT_32'])
            epoch_loss += loss.item()
            progress_bar.set_postfix({"all": epoch_loss / (step + 1)}) 
    
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch:3d}, Learning Rate: {current_lr:.2e}')
        test_ct_32(modules, val_loader, EMA['CT_32'].get() if ('CT_32' in EMA) else MODEL['CT_32'], save_flag=True, epoch=epoch)
        

logging.basicConfig(filename=log_path.format('32-fast'), filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
modules = DIM_Modules()


# test_ct_32(modules, test_loader, MODEL['CT_32'], save_flag=False, epoch='test')
train_ct_32(modules, train_loader, MODEL['CT_32'])
