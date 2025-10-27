import os
import time
import math
import torch
import logging
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from monai.data import DataLoader
from monai.metrics import PSNRMetric
from generative.metrics import SSIMMetric
from generative.losses import PerceptualLoss
from generative.networks.nets import DiffusionModelUNet


from dataset import *
from utils import *
##### 
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
torch.backends.cudnn.benchmark = True


batch_size, batch_size_val = 32, 16
num_workers = 32
learning_rate = 2e-4
ema_start = 5
ema_rate = 0.9995

sparsity = 32
resol_levels = [32]  #####
MINI_BATCH_SIZE = {'32': 32, '64': 32, '128': 32, '256': 16}   #####
VAL_LOSSES = {'32': [], '64': [], '128': [], '256': []}



RADON = {}
RADON['32x256'] = fanbeam_gen(sparsity=(32, 256), img_size=256, bias=0)
RADON['64x256'] = fanbeam_gen(sparsity=(64, 256), img_size=256, bias=0)
RADON['128x256'] = fanbeam_gen(sparsity=(128, 256), img_size=256, bias=0)
RADON['256x256'] = fanbeam_gen(sparsity=(256, 256), img_size=256, bias=0)
RADON['32x16'] =  fanbeam_gen(sparsity=(32, 32), img_size=16, bias=0, pixel_spacing=1.4285/16, det_spacing=1.2858*16)
RADON['64x32'] = fanbeam_gen(sparsity=(64, 64), img_size=32, bias=0, pixel_spacing=1.4285/8, det_spacing=1.2858*8)
RADON['128x64'] = fanbeam_gen(sparsity=(128, 128), img_size=64, bias=0, pixel_spacing=1.4285/4, det_spacing=1.2858*4)
RADON['256x128'] = fanbeam_gen(sparsity=(256, 256), img_size=128, bias=0, pixel_spacing=1.4285/2, det_spacing=1.2858*2)
RADON['512x256'] = fanbeam_gen(sparsity=(512, 512), img_size=256, bias=0, pixel_spacing=1.4285, det_spacing=1.2858)

RADON['32x32'] = fanbeam_gen(sparsity=(32, 32), img_size=32, bias=0, pixel_spacing=1.4285/8, det_spacing=1.2858*8)
RADON['32x64'] = fanbeam_gen(sparsity=(32, 32), img_size=64, bias=0, pixel_spacing=1.4285/4, det_spacing=1.2858*4)
RADON['32x128'] = fanbeam_gen(sparsity=(32, 32), img_size=128, bias=0, pixel_spacing=1.4285/2, det_spacing=1.2858*2)


psnr = PSNRMetric(max_val=1.)
ssim = SSIMMetric(spatial_dims=2, data_range=1.)
lpips = PerceptualLoss(spatial_dims=2, network_type="alex").cuda()

train_loader = DataLoader(CTDataset('/workspace/DATASET/SVCT/ldct/train_set', mode='train'), batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True)
val_loader = DataLoader(CTDataset('/workspace/DATASET/SVCT/ldct/val_set', mode='val'), batch_size=batch_size_val, shuffle=False, num_workers=num_workers, persistent_workers=True)
head_loader = DataLoader(CTDataset('/workspace/DATASET/SVCT/ldct/n_set', mode='val'), batch_size=batch_size_val, shuffle=False, num_workers=num_workers, persistent_workers=True)
overfitting_loader = DataLoader(CTDataset('/workspace/DATASET/SVCT/ldct/train_set', mode='overfitting'), batch_size=batch_size_val, shuffle=False, num_workers=num_workers, persistent_workers=True)
test_loader = DataLoader(CTDataset('/workspace/DATASET/SVCT/ldct/test_set', mode='val'), batch_size=batch_size_val, shuffle=False, num_workers=num_workers, persistent_workers=True)




MODEL, EMA = {}, {}

class MultiModelWrapper(nn.Module):
    def __init__(self, base_model_class, num_models, resolution, **base_model_kwargs):
        super(MultiModelWrapper, self).__init__()
        self.num_models = num_models
        self.resol = resolution
        self.models = nn.ModuleList([
            base_model_class(**base_model_kwargs)
            for _ in range(num_models)
        ])

    def __getitem__(self, idx):
        return self.models[idx]
    
    def __len__(self):
        return len(self.models)
    #####
    def forward(self, ct, sino, radon):
        CT_OUTPUT = []
        if self.resol in [32,64,128]:
            store, store2, store3, indices_to_remove = (2,4,6,9,12,16), (4,9,16), (4,9,16), [1,2,7,8,13,14]
        elif self.resol in [256]:
            # store, indices_to_remove = (2,4,8,10,12,14,16,18,20,22,24,26), []
            store, indices_to_remove = (2,4,6,8,10), []
        
        with torch.no_grad(): ct = F.interpolate(ct, scale_factor=2, mode='bicubic')
        for i, model in enumerate(self.models):
            with torch.no_grad():
                ct_wls = WLS_fast(sino, ct, store, radon)
                mask = torch.ones(ct_wls.size(1), dtype=torch.bool)
                mask[indices_to_remove] = False
                ct_wls = ct_wls[:,mask]
                ct_wls_diff = ct_wls - ct.repeat(1, ct_wls.shape[1],1,1)
                ct_input = torch.cat([ct_wls, ct_wls_diff, ct], dim=1)
                timesteps = torch.full((ct_input.shape[0],), 0).long().cuda()
            
            ct_output = model(x=ct_input, timesteps=timesteps) + ct
            ct = ct_output
            CT_OUTPUT.append(ct_output)
        
        return CT_OUTPUT




MODEL[f'CT_{resol_levels[0]}'] = MultiModelWrapper(
    base_model_class=DiffusionModelUNet,
    num_models=2,
    resolution=32,
    spatial_dims=2, 
    num_res_blocks=2,
    in_channels=25,
    out_channels=1,
    num_channels=(96, 96),
    attention_levels=(True, True),
    num_head_channels=(96, 96),
).cuda()



MODEL[f'CT_{resol_levels[1]}'] = MultiModelWrapper(
    base_model_class=DiffusionModelUNet,
    num_models=3,
    resolution=64,
    spatial_dims=2, 
    num_res_blocks=2,
    in_channels=25,
    out_channels=1,
    num_channels=(64, 64),
    attention_levels=(False, True),
    num_head_channels=(0, 64),
).cuda()



MODEL[f'CT_{resol_levels[2]}'] = MultiModelWrapper(
    base_model_class=DiffusionModelUNet,
    num_models=4,
    resolution=128,
    spatial_dims=2, 
    num_res_blocks=2,
    in_channels=25,
    out_channels=1,
    num_channels=(48, 48, 48),
    norm_num_groups = 24,
    attention_levels=(False, False, False),
).cuda()


#####
MODEL[f'CT_{resol_levels[3]}'] = MultiModelWrapper(
    base_model_class=DiffusionModelUNet,
    num_models=4,
    resolution=256,
    spatial_dims=2, 
    num_res_blocks=2,
    in_channels=11,
    out_channels=1,
    num_channels=(32, 32, 64, 64),
    norm_num_groups = 32,
    attention_levels=(False, False, False, False),
).cuda()




def correct_ct_final(ct_input, sino, radon, store, lambda_reg):
    ct_wls = WLS_fast_ridge(sino, ct_input, store, radon, lambda_reg)
    return ct_wls

def loss_cal(ct_pred, ct_ref, radon):
    loss_CT = F.l1_loss(ct_pred, ct_ref)
    loss_SINO = F.l1_loss(radon.filter_sinogram(radon.forward(ct_pred), filter_name='ramp'), radon.filter_sinogram(radon.forward(ct_ref), filter_name='ramp'))
    return loss_CT + loss_SINO*10


CT, SINO = {}, {}
def pre_cal(ct_full):
    SINO['32x256'] = RADON['32x256'].forward(ct_full)
    SINO['64x256'] = RADON['64x256'].forward(ct_full)
    SINO['128x256'] = RADON['128x256'].forward(ct_full)
    SINO['256x256'] = RADON['256x256'].forward(ct_full)
    SINO['512x256'] = RADON['512x256'].forward(ct_full)


    SINO['32x16'] = torch.cat([F.adaptive_avg_pool2d(SINO['32x256'], (32,16)) / 16, F.adaptive_max_pool2d(SINO['32x256'], (32,16)) / 16, -F.adaptive_max_pool2d(-SINO['32x256'], (32,16)) / 16], dim=1)
    SINO['32x32'] = torch.cat([F.adaptive_avg_pool2d(SINO['32x256'], (32,32)) / 8, F.adaptive_max_pool2d(SINO['32x256'], (32,32)) / 8, -F.adaptive_max_pool2d(-SINO['32x256'], (32,32)) / 8], dim=1)
    SINO['64x32'] = torch.cat([F.adaptive_avg_pool2d(SINO['64x256'], (64,32)) / 8, F.adaptive_max_pool2d(SINO['64x256'], (64,32)) / 8, -F.adaptive_max_pool2d(-SINO['64x256'], (64,32)) / 8], dim=1)
    SINO['32x64'] = torch.cat([F.adaptive_avg_pool2d(SINO['32x256'], (32,64)) / 4, F.adaptive_max_pool2d(SINO['32x256'], (32,64)) / 4, -F.adaptive_max_pool2d(-SINO['32x256'], (32,64)) / 4], dim=1)
    SINO['64x64'] = torch.cat([F.adaptive_avg_pool2d(SINO['64x256'], (64,64)) / 4, F.adaptive_max_pool2d(SINO['64x256'], (64,64)) / 4, -F.adaptive_max_pool2d(-SINO['64x256'], (64,64)) / 4], dim=1)
    SINO['128x64'] = torch.cat([F.adaptive_avg_pool2d(SINO['128x256'], (128,64)) / 4, F.adaptive_max_pool2d(SINO['128x256'], (128,64)) / 4, -F.adaptive_max_pool2d(-SINO['128x256'], (128,64)) / 4], dim=1)
    SINO['32x128'] = torch.cat([F.adaptive_avg_pool2d(SINO['32x256'], (32,128)) / 2, F.adaptive_max_pool2d(SINO['32x256'], (32,128)) / 2, -F.adaptive_max_pool2d(-SINO['32x256'], (32,128)) / 2], dim=1)
    SINO['256x128'] = torch.cat([F.adaptive_avg_pool2d(SINO['256x256'], (256,128)) / 2, F.adaptive_max_pool2d(SINO['256x256'], (256,128)) / 2, -F.adaptive_max_pool2d(-SINO['256x256'], (256,128)) / 2], dim=1)


    CT['32x32'] = torch.cat([F.adaptive_avg_pool2d(ct_full, (32,32)), F.adaptive_max_pool2d(ct_full, (32,32)), -F.adaptive_max_pool2d(-ct_full, (32,32))], dim=1)
    CT['64x64'] = torch.cat([F.adaptive_avg_pool2d(ct_full, (64,64)), F.adaptive_max_pool2d(ct_full, (64,64)), -F.adaptive_max_pool2d(-ct_full, (64,64))], dim=1)
    CT['128x128'] = torch.cat([F.adaptive_avg_pool2d(ct_full, (128,128)), F.adaptive_max_pool2d(ct_full, (128,128)), -F.adaptive_max_pool2d(-ct_full, (128,128))], dim=1)
    CT['256x256'] = ct_full

    CT['16x16_fbp'] = recon(SINO['32x16'][:,0:1], RADON['32x16'])
    return CT['16x16_fbp']



    




def test_ct(test_loader, test_models, save_flag=False, epoch='test'):
    with torch.no_grad():
        for test_model in test_models: test_model.eval()
        EVAL = {}
        for resolution in resol_levels:
            EVAL[f'SSIM_{resolution}'], EVAL[f'PSNR_{resolution}'], EVAL[f'LPIPS_{resolution}'], EVAL[f'L1_{resolution}'], EVAL[f'L2_{resolution}'] = [], [], [], [], []
        
        start = time.time()
        for _, batch in enumerate(test_loader, start=0):
            ct_full = pre_process(batch, val_flag=True)
            
            ct_input = pre_cal(ct_full)
            for i, resolution in enumerate(resol_levels):
                CT_OUTPUT = test_models[i](ct=ct_input, sino=SINO[f'{sparsity}x{resolution}'], radon=RADON[f'{sparsity}x{resolution}'])
                
                ct_input, pred = CT_OUTPUT[-1], CT_OUTPUT[-1]
                if resolution == 256: pred = correct_ct_final(ct_input=ct_input, sino=SINO[f'{sparsity}x{resolution}'], radon=RADON[f'{sparsity}x{resolution}'], store=[50], lambda_reg=0)
                EVAL = evaluate(pred, CT[f'{resolution}x{resolution}'][:,0:1], ssim, psnr, lpips, EVAL, resolution)


        end_time = time.time()
        print(f"inference time: {end_time - start} s")
        for resolution in resol_levels:
            ssim_mean, ssim_std = np.mean(EVAL[f'SSIM_{resolution}']), np.std(EVAL[f'SSIM_{resolution}'])
            psnr_mean, psnr_std = np.mean(EVAL[f'PSNR_{resolution}']), np.std(EVAL[f'PSNR_{resolution}'])
            lpips_mean, lpips_std = np.mean(EVAL[f'LPIPS_{resolution}']), np.std(EVAL[f'LPIPS_{resolution}'])
            l1_mean, l1_std = np.mean(EVAL[f'L1_{resolution}']), np.std(EVAL[f'L1_{resolution}'])
            l2_mean, l2_std = np.mean(EVAL[f'L2_{resolution}']), np.std(EVAL[f'L2_{resolution}'])

            text = f"epoch: {epoch}, resol: {resolution} ssim mean: {round(ssim_mean,6)}, ssim std: {round(ssim_std,6)}; psnr mean: {round(psnr_mean,6)}, psnr std: {round(psnr_std,6)}; lpips mean: {round(lpips_mean,6)}, lpips std: {round(lpips_std,6)}, l1 mean: {round(l1_mean,6) }, l1 std: {round(l1_std,6)}; l2 mean: {round(l2_mean,6)}, l2 std: {round(l2_std,6)}"
            print(text)
            logging.info(text)

        if save_flag: 
            VAL_LOSSES[f'{resolution}'].append(ssim_mean)
            if ssim_mean == np.max(VAL_LOSSES[f'{resolution}']) and (f'CT_{resolution}' in EMA):
                torch.save({'model': test_model.state_dict()}, model_save_path+f'{epoch}-ct-split.pth')
    for test_model in test_models: test_model.train()






def train_ct(train_loader, train_model):
    epoch_start = 0
    epoch_end = 90
    optimizer = torch.optim.AdamW(params=train_model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epoch_end-15,epoch_end-5], gamma=0.25)

    for epoch in range(epoch_start, epoch_end):
        epoch_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=120)
        progress_bar.set_description(f"Epoch {epoch}")
        if epoch == ema_start or bool(epoch_start): 
            EMA[f'CT_{resol_levels[-1]}'] = ModelEMA(train_model, decay=ema_rate)
        for step, batch in progress_bar:
            optimizer.zero_grad(set_to_none=True)


            with torch.no_grad():
                ct_full = pre_process(batch, val_flag=False)
                ct_input = pre_cal(ct_full)
                for _, resolution in enumerate(resol_levels[:-1]):
                    RANGE=[0, ct_full.shape[0]]
                    CT_OUTPUT = MODEL[f'CT_{resolution}'](ct=ct_input[RANGE[0]:RANGE[1]], sino=SINO[f'{sparsity}x{resolution}'][RANGE[0]:RANGE[1]], radon=RADON[f'{sparsity}x{resolution}'])
                    ct_input = CT_OUTPUT[-1]

            resolution = resol_levels[-1]
            for i in range(math.ceil(ct_full.shape[0]/MINI_BATCH_SIZE[f'{resolution}'])):
                loss = 0
                optimizer.zero_grad(set_to_none=True)
                RANGE=[i*MINI_BATCH_SIZE[f'{resolution}'], i*MINI_BATCH_SIZE[f'{resolution}']+(min(ct_full.shape[0]-i*MINI_BATCH_SIZE[f'{resolution}'], MINI_BATCH_SIZE[f'{resolution}']))]
                CT_OUTPUT = train_model(ct=ct_input[RANGE[0]:RANGE[1]], sino=SINO[f'{sparsity}x{resolution}'][RANGE[0]:RANGE[1]], radon=RADON[f'{sparsity}x{resolution}'])
                for j in range(len(train_model)):
                    loss = loss + loss_cal(ct_pred=CT_OUTPUT[j], ct_ref=CT[f'{resolution}x{resolution}'][RANGE[0]:RANGE[1],0:1], radon=RADON[f'{int(2*sparsity/resol_levels[0]*resolution)}x{resolution}'])
                loss.backward()
                optimizer.step()

                if f'CT_{resolution}' in EMA: EMA[f'CT_{resolution}'].update(train_model)
                epoch_loss += loss.item()
                progress_bar.set_postfix({"all": epoch_loss / (step*batch_size//MINI_BATCH_SIZE[f'{resolution}'] + (i+1))}) 

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch:3d}, Learning Rate: {current_lr:.2e}')

        #####
        test_ct(val_loader, [MODEL['CT_32'], MODEL['CT_64'], MODEL['CT_128'], EMA[f'CT_{resolution}'].get() if (f'CT_{resolution}' in EMA) else MODEL[f'CT_{resolution}']], save_flag=True, epoch=epoch)
        test_ct(head_loader, [MODEL['CT_32'], MODEL['CT_64'], MODEL['CT_128'], EMA[f'CT_{resolution}'].get() if (f'CT_{resolution}' in EMA) else MODEL[f'CT_{resolution}']], save_flag=True, epoch=epoch)





#####
log_path = f'/workspace/PROJECT/SVCT/WLS_{sparsity}/test_log/{resol_levels[-1]}-split.log'
model_save_path = f'/workspace/MODEL/SVCT/{sparsity}/WLS_split/{resol_levels[-1]}/'

logging.basicConfig(filename=log_path, filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

model_path = '/workspace/MODEL/SVCT/32/WLS_split/32/89-ct-split.pth'
state_dict = torch.load(model_path)['model']
MODEL[f'CT_{resol_levels[0]}'].load_state_dict(state_dict)


model_path = '/workspace/MODEL/SVCT/32/WLS_split/64/89-ct-split.pth'
state_dict = torch.load(model_path)['model']
MODEL[f'CT_{resol_levels[1]}'].load_state_dict(state_dict)


#####
model_path = '/workspace/MODEL/SVCT/32/WLS_split/128/89-ct-split.pth'
state_dict = torch.load(model_path)['model']
MODEL[f'CT_{resol_levels[2]}'].load_state_dict(state_dict)


# model_path = '/workspace/MODEL/SVCT/32/WLS_split/256/46-ct-split.pth'
# state_dict = torch.load(model_path)['model']
# MODEL[f'CT_{resol_levels[3]}'].load_state_dict(state_dict)


# test_ct(head_loader, [MODEL['CT_32'], MODEL['CT_64'], MODEL['CT_128'], MODEL['CT_256']], save_flag=False, epoch='head_set')

train_ct(train_loader, MODEL[f'CT_{resol_levels[-1]}'])
