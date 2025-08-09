import os
import time
import math
import torch
import logging
import numpy as np
import contextlib
from tqdm import tqdm
import torch.nn.functional as F
from monai.data import DataLoader
from monai.metrics import PSNRMetric
from generative.metrics import SSIMMetric
from generative.losses import PerceptualLoss
from generative.networks.nets import DiffusionModelUNet
from dataset import *
from utils import *
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
torch.backends.cudnn.benchmark = True


batch_size = 4 
mini_batch_size_32 = 4 
mini_batch_size_64 = 4 
mini_batch_size_128 = 4 
mini_batch_size_256 = 4
num_workers = 16
learning_rate = 2e-4
ema_rate = 0.9995



start_32, end_32 = 0, 90
start_64, end_64 = 90, 180
start_128, end_128 = 180, 270
start_256, end_256 = 270, 360

VAL_LOSSES_CT_64, VAL_LOSSES_CT_128, VAL_LOSSES_CT_256 = [], [], []


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

train_loader = DataLoader(CTDataset('/workspace/DATASET/AAPM/train_set', mode='train'), batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True)
val_loader = DataLoader(CTDataset('/workspace/DATASET/AAPM/val_set', mode='val'), batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True)
test_loader = DataLoader(CTDataset('/workspace/DATASET/AAPM/test_set', mode='val'), batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True)

log_path = '/DIM/{}.log'
model_save_path_256 = '/DIM/{}.pth'
logging.basicConfig(filename=log_path.format('test'), filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


MODEL, EMA = {}, {}

MODEL['CT_32'] = DiffusionModelUNet(spatial_dims=2, num_res_blocks=2,
    in_channels=49,
    out_channels=1,
    num_channels=(128, 128, 256),
    attention_levels=(False, True, True),
    num_head_channels=(0, 128, 256),
).cuda()

MODEL['CT_64'] = DiffusionModelUNet(spatial_dims=2, num_res_blocks=2,
    in_channels=49,
    out_channels=1,
    num_channels=(96, 128, 128, 256),
    attention_levels=(False, False, True, True),
    num_head_channels=(0, 0, 128, 256),
).cuda()

MODEL['CT_128'] = DiffusionModelUNet(spatial_dims=2, num_res_blocks=2,
    in_channels=49,
    out_channels=1,
    num_channels=(96, 96, 128, 128, 256),
    attention_levels=(False, False, False, True, True),
    num_head_channels=(0, 0, 0, 128, 256),
).cuda()


MODEL['CT_256'] = DiffusionModelUNet(spatial_dims=2, num_res_blocks=2,
    in_channels=49,
    out_channels=1,
    num_channels=(64, 96, 96, 128, 128, 256),
    attention_levels=(False, False, False, False, True, True),
    num_head_channels=(0, 0, 0, 0, 128, 256),
).cuda()


class Fusion_Modules():
    def __init__(self):
        self.CT, self.SINO = {}, {}


    def pre_cal(self, ct_full):
        self.b, _, _, _ = ct_full.shape

        self.SINO['32x256'] = RADON['32x256'].forward(ct_full)
        self.SINO['64x256'] = RADON['64x256'].forward(ct_full)
        self.SINO['128x256'] = RADON['128x256'].forward(ct_full)
        self.SINO['256x256'] = RADON['256x256'].forward(ct_full)
        self.SINO['512x256'] = RADON['512x256'].forward(ct_full)


        self.SINO['32x16'] = torch.cat([F.adaptive_avg_pool2d(self.SINO['32x256'], (32,16)) / 16, F.adaptive_max_pool2d(self.SINO['32x256'], (32,16)) / 16, 1-F.adaptive_max_pool2d(1-self.SINO['32x256'], (32,16)) / 16], dim=1)
        self.SINO['32x32'] = torch.cat([F.adaptive_avg_pool2d(self.SINO['32x256'], (32,32)) / 8, F.adaptive_max_pool2d(self.SINO['32x256'], (32,32)) / 8, 1-F.adaptive_max_pool2d(1-self.SINO['32x256'], (32,32)) / 8], dim=1)
        self.SINO['64x32'] = torch.cat([F.adaptive_avg_pool2d(self.SINO['64x256'], (64,32)) / 8, F.adaptive_max_pool2d(self.SINO['64x256'], (64,32)) / 8, 1-F.adaptive_max_pool2d(1-self.SINO['64x256'], (64,32)) / 8], dim=1)
        self.SINO['32x64'] = torch.cat([F.adaptive_avg_pool2d(self.SINO['32x256'], (32,64)) / 4, F.adaptive_max_pool2d(self.SINO['32x256'], (32,64)) / 4, 1-F.adaptive_max_pool2d(1-self.SINO['32x256'], (32,64)) / 4], dim=1)
        self.SINO['64x64'] = torch.cat([F.adaptive_avg_pool2d(self.SINO['64x256'], (64,64)) / 4, F.adaptive_max_pool2d(self.SINO['64x256'], (64,64)) / 4, 1-F.adaptive_max_pool2d(1-self.SINO['64x256'], (64,64)) / 4], dim=1)
        self.SINO['128x64'] = torch.cat([F.adaptive_avg_pool2d(self.SINO['128x256'], (128,64)) / 4, F.adaptive_max_pool2d(self.SINO['128x256'], (128,64)) / 4, 1-F.adaptive_max_pool2d(1-self.SINO['128x256'], (128,64)) / 4], dim=1)
        self.SINO['32x128'] = torch.cat([F.adaptive_avg_pool2d(self.SINO['32x256'], (32,128)) / 2, F.adaptive_max_pool2d(self.SINO['32x256'], (32,128)) / 2, 1-F.adaptive_max_pool2d(1-self.SINO['32x256'], (32,128)) / 2], dim=1)
        self.SINO['256x128'] = torch.cat([F.adaptive_avg_pool2d(self.SINO['256x256'], (256,128)) / 2, F.adaptive_max_pool2d(self.SINO['256x256'], (256,128)) / 2, 1-F.adaptive_max_pool2d(1-self.SINO['256x256'], (256,128)) / 2], dim=1)


        self.CT['32x32'] = torch.cat([F.adaptive_avg_pool2d(ct_full, (32,32)), F.adaptive_max_pool2d(ct_full, (32,32)), 1-F.adaptive_max_pool2d(1-ct_full, (32,32))], dim=1)
        self.CT['64x64'] = torch.cat([F.adaptive_avg_pool2d(ct_full, (64,64)), F.adaptive_max_pool2d(ct_full, (64,64)), 1-F.adaptive_max_pool2d(1-ct_full, (64,64))], dim=1)
        self.CT['128x128'] = torch.cat([F.adaptive_avg_pool2d(ct_full, (128,128)), F.adaptive_max_pool2d(ct_full, (128,128)), 1-F.adaptive_max_pool2d(1-ct_full, (128,128))], dim=1)
        self.CT['256x256'] = ct_full


        


    def predict_sino_32(self, numStore1=(2,4,6,9,12,16,20,25,30,36,42,50)):
        self.CT['16x16_fbp'] = recon(self.SINO['32x16'][:,0].unsqueeze(1), RADON['32x16'])
        self.CT['32x32_fbp_interp'] = F.interpolate(self.CT['16x16_fbp'], (32,32), mode='bicubic')
       
        self.CT['32x32_wls_all'] = WLS_ridge(self.SINO['32x32'], self.CT['32x32_fbp_interp'], numStore1, RADON['32x32'])
        indices_to_remove = [1,2,7,8,13,14,19,20,25,26,31,32]
        mask = torch.ones(self.CT['32x32_wls_all'].size(1), dtype=torch.bool)
        mask[indices_to_remove] = False
        self.CT['32x32_wls'] = self.CT['32x32_wls_all'][:,mask]
        
        self.CT['32x32_wls_diff'] = self.CT['32x32_wls'] - self.CT['32x32_fbp_interp'].repeat(1,24,1,1)
        self.CT['32x32_output'] = torch.ones_like(self.CT['32x32_fbp_interp'])
        return self.CT['32x32_wls'], self.CT['32x32']
    

    def predict_ct_32(self, ct_model, RANGE):
        self.timesteps = torch.full((RANGE[1]-RANGE[0],), 0).long().cuda()
        self.CT['32x32_input'] = torch.cat([self.CT['32x32_wls'][RANGE[0]:RANGE[1]], self.CT['32x32_wls_diff'][RANGE[0]:RANGE[1]], self.CT['32x32_fbp_interp'][RANGE[0]:RANGE[1]]], dim=1) 
        self.CT['32x32_output_mini'] = ct_model(x = self.CT['32x32_input'], timesteps = self.timesteps) + self.CT['32x32_fbp_interp'][RANGE[0]:RANGE[1]]
        self.CT['32x32_output_mini'] = self.CT['32x32_output_mini'] 
        self.CT['32x32_output'][RANGE[0]:RANGE[1]] = self.CT['32x32_output_mini']

        return self.CT['32x32_output_mini'], self.CT['32x32'][RANGE[0]:RANGE[1], 0].unsqueeze(1)


    def predict_sino_64(self, numStore1=(2,4,6,9,12,16,20,25,30,36,42,50)): 
        self.CT['32x32_fbp'] = recon(self.SINO['64x32'][:,0].unsqueeze(1), RADON['64x32'])
        self.CT['64x64_fbp_interp'] = F.interpolate(self.CT['32x32_fbp'], (64,64), mode='bicubic') 

        self.CT['64x64_wls_all'] = WLS_ridge(self.SINO['32x64'], self.CT['64x64_fbp_interp'], numStore1, RADON['32x64'])
        indices_to_remove = [1,2,7,8,13,14,19,20,25,26,31,32]
        mask = torch.ones(self.CT['64x64_wls_all'].size(1), dtype=torch.bool)
        mask[indices_to_remove] = False
        self.CT['64x64_wls'] = self.CT['64x64_wls_all'][:,mask]

        self.CT['64x64_wls_diff'] = self.CT['64x64_wls'] - self.CT['64x64_fbp_interp'].repeat(1,24,1,1)
        self.CT['64x64_output'] = torch.ones_like(self.CT['64x64_fbp_interp'])
        return self.CT['64x64_wls'], self.CT['64x64']
    

    def predict_ct_64(self, ct_model, RANGE):
        self.timesteps = torch.full((RANGE[1]-RANGE[0],), 0).long().cuda()
        self.CT['64x64_input'] = torch.cat([self.CT['64x64_wls'][RANGE[0]:RANGE[1]], self.CT['64x64_wls_diff'][RANGE[0]:RANGE[1]], self.CT['64x64_fbp_interp'][RANGE[0]:RANGE[1]]], dim=1) 
        self.CT['64x64_output_mini'] = ct_model(x = self.CT['64x64_input'], timesteps = self.timesteps) + self.CT['64x64_fbp_interp'][RANGE[0]:RANGE[1]]
        self.CT['64x64_output_mini'] = self.CT['64x64_output_mini'] 
        self.CT['64x64_output'][RANGE[0]:RANGE[1]] = self.CT['64x64_output_mini']
        return self.CT['64x64_output_mini'], self.CT['64x64'][RANGE[0]:RANGE[1], 0].unsqueeze(1)


    def predict_sino_128(self, numStore1=(2,4,6,9,12,16,20,25,30,36,42,50)):
        self.CT['128x128_fbp_interp'] = F.interpolate(self.CT['64x64_output'], (128,128), mode='bicubic')
        self.CT['128x128_wls_all'] = WLS_ridge(self.SINO['32x128'], self.CT['128x128_fbp_interp'], numStore1, RADON['32x128'])

        indices_to_remove = [1,2,7,8,13,14,19,20,25,26,31,32]
        mask = torch.ones(self.CT['128x128_wls_all'].size(1), dtype=torch.bool)
        mask[indices_to_remove] = False
        self.CT['128x128_wls'] = self.CT['128x128_wls_all'][:,mask]

        self.CT['128x128_wls_diff'] = self.CT['128x128_wls'] - self.CT['128x128_fbp_interp'].repeat(1,24,1,1)
        self.CT['128x128_output'] = torch.ones_like(self.CT['128x128_fbp_interp'])
        return self.CT['128x128_wls'], self.CT['128x128_wls_diff']


    def predict_ct_128(self, ct_model, RANGE):
        self.timesteps = torch.full((RANGE[1]-RANGE[0],), 0).long().cuda()
        self.CT['128x128_input'] = torch.cat([self.CT['128x128_wls'][RANGE[0]:RANGE[1]], self.CT['128x128_wls_diff'][RANGE[0]:RANGE[1]], self.CT['128x128_fbp_interp'][RANGE[0]:RANGE[1]]], dim=1) 
        self.CT['128x128_output_mini'] = ct_model(x = self.CT['128x128_input'], timesteps = self.timesteps) + self.CT['128x128_fbp_interp'][RANGE[0]:RANGE[1]]
        self.CT['128x128_output_mini'] = self.CT['128x128_output_mini']
        self.CT['128x128_output'][RANGE[0]:RANGE[1]] = self.CT['128x128_output_mini']
        return self.CT['128x128_output_mini'], self.CT['128x128'][RANGE[0]:RANGE[1], 0].unsqueeze(1)
    

    def predict_sino_256(self, numStore=(2,4,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50)):
        self.CT['256x256_fbp_interp'] = F.interpolate(self.CT['128x128_output'], (256,256), mode='bicubic') 
        self.CT['256x256_wls'] = WLS_ridge(self.SINO['32x256'], self.CT['256x256_fbp_interp'], numStore, RADON['32x256'])

        self.CT['256x256_wls_diff'] = self.CT['256x256_wls'] - self.CT['256x256_fbp_interp']
        self.CT['256x256_output'] = torch.ones_like(self.CT['256x256_fbp_interp'])
        return self.CT['256x256_wls'], self.CT['256x256_wls_diff']

    
    def predict_ct_256(self, ct_model, RANGE):
        self.timesteps = torch.full((RANGE[1]-RANGE[0],), 0).long().cuda()
        self.CT['256x256_input'] = torch.cat([self.CT['256x256_wls'][RANGE[0]:RANGE[1]], self.CT['256x256_wls_diff'][RANGE[0]:RANGE[1]], self.CT['256x256_fbp_interp'][RANGE[0]:RANGE[1]]], dim=1) 
        self.CT['256x256_output_mini'] = ct_model(x = self.CT['256x256_input'], timesteps = self.timesteps) + self.CT['256x256_fbp_interp'][RANGE[0]:RANGE[1]]
        self.CT['256x256_output_mini'] = self.CT['256x256_output_mini'] 
        self.CT['256x256_output'][RANGE[0]:RANGE[1]] = self.CT['256x256_output_mini']
        return self.CT['256x256_output_mini'], self.CT['256x256'][RANGE[0]:RANGE[1], 0].unsqueeze(1)


    def correct_ct_256(self, num_Store, lambda_reg):
        if num_Store[0] == 0:
            return self.CT['256x256_output'], self.CT['256x256']
        else:
            self.CT['256x256_output_wls'] = WLS_ridge(self.SINO['32x256'], self.CT['256x256_output'], num_Store, RADON['32x256'], lambda_reg)
            return self.CT['256x256_output_wls'], self.CT['256x256']



    def loss_cal_ct_32(self, ref):
        loss_CT = F.l1_loss(self.CT['32x32_output'], ref)
        loss_SINO = F.l1_loss(RADON['64x32'].filter_sinogram(RADON['64x32'].forward(self.CT['32x32_output']), filter_name='ramp'), RADON['64x32'].filter_sinogram(RADON['64x32'].forward(ref), filter_name='ramp'))
        return [loss_CT, loss_SINO*10]
    
    def loss_cal_ct_64(self, ref):
        loss_CT = F.l1_loss(self.CT['64x64_output_mini'], ref)
        loss_SINO = F.l1_loss(RADON['128x64'].filter_sinogram(RADON['128x64'].forward(self.CT['64x64_output_mini']), filter_name='ramp'), RADON['128x64'].filter_sinogram(RADON['128x64'].forward(ref), filter_name='ramp'))
        return [loss_CT, loss_SINO*10]
    
    def loss_cal_ct_128(self, ref):
        loss_CT = F.l1_loss(self.CT['128x128_output_mini'], ref)
        loss_SINO = F.l1_loss(RADON['256x128'].filter_sinogram(RADON['256x128'].forward(self.CT['128x128_output_mini']), filter_name='ramp'), RADON['256x128'].filter_sinogram(RADON['256x128'].forward(ref), filter_name='ramp'))
        return [loss_CT, loss_SINO*10]

    def loss_cal_ct_256(self, ref):
        loss_CT = F.l1_loss(self.CT['256x256_output_mini'], ref)
        loss_SINO = F.l1_loss(RADON['512x256'].filter_sinogram(RADON['512x256'].forward(self.CT['256x256_output_mini']), filter_name='ramp'), RADON['512x256'].filter_sinogram(RADON['512x256'].forward(ref), filter_name='ramp'))
        return [loss_CT, loss_SINO*10]



def test_ct_256(modules, test_loader, test_model_32, test_model_64, test_model_128, test_model_256, save_flag=False, epoch=0):
    with torch.no_grad():
        test_model_32.eval()
        test_model_64.eval()
        test_model_128.eval()
        test_model_256.eval()

        SSIM_32, PSNR_32, LPIPS_32, L1_32, L2_32 = [], [], [], [], []
        SSIM_64, PSNR_64, LPIPS_64, L1_64, L2_64 = [], [], [], [], []
        SSIM_128, PSNR_128, LPIPS_128, L1_128, L2_128 = [], [], [], [], []
        SSIM_256, PSNR_256, LPIPS_256, L1_256, L2_256 = [], [], [], [], []

        start = time.time()
        for val_step, batch in enumerate(test_loader, start=0):
            ct_full = pre_process(batch, val_flag=True)
            
            modules.pre_cal(ct_full)
            modules.predict_sino_32()
            for i in range(math.ceil(ct_full.shape[0]/mini_batch_size_32)):
                RANGE=[i*mini_batch_size_32, i*mini_batch_size_32+(min(ct_full.shape[0]-i*mini_batch_size_32, mini_batch_size_32))]
                ct_pred_32, ct_ref_32 = modules.predict_ct_32(test_model_32, RANGE)

            if epoch >= start_64:
                modules.predict_sino_64()
                for i in range(math.ceil(ct_full.shape[0]/mini_batch_size_64)):
                    RANGE=[i*mini_batch_size_64, i*mini_batch_size_64+(min(ct_full.shape[0]-i*mini_batch_size_64, mini_batch_size_64))]
                    ct_pred_64, ct_ref_64 = modules.predict_ct_64(test_model_64, RANGE)

            if epoch >= start_128:
                modules.predict_sino_128()
                for i in range(math.ceil(ct_full.shape[0]/mini_batch_size_128)):
                    RANGE=[i*mini_batch_size_128, i*mini_batch_size_128+(min(ct_full.shape[0]-i*mini_batch_size_128, mini_batch_size_128))]
                    ct_pred_128, ct_ref_128 = modules.predict_ct_128(test_model_128, RANGE)

            if epoch >= start_256:
                modules.predict_sino_256()
                for i in range(math.ceil(ct_full.shape[0]/mini_batch_size_256)):
                    RANGE=[i*mini_batch_size_256, i*mini_batch_size_256+(min(ct_full.shape[0]-i*mini_batch_size_256, mini_batch_size_256))]
                    modules.predict_ct_256(test_model_256, RANGE)
                
                ct_pred_256, ct_ref_256 = modules.correct_ct_256(num_Store=[50], lambda_reg=0)


            for i in range(math.ceil(ct_full.shape[0]/mini_batch_size_256)):
                RANGE=[i*mini_batch_size_256, i*mini_batch_size_256+(min(ct_full.shape[0]-i*mini_batch_size_256, mini_batch_size_256))]



                ct_pred, ct_ref = torch.clip(ct_pred_32, min=0, max=1), torch.clip(ct_ref_32, min=0, max=1)
                SSIM_32.append(ssim(ct_pred, ct_ref).mean().item())
                PSNR_32.append(psnr(ct_pred, ct_ref).mean().item())
                LPIPS_32.append(lpips(ct_pred, ct_ref).mean().item())
                L1_32.append(F.l1_loss(ct_pred, ct_ref).mean().item())
                L2_32.append(F.mse_loss(ct_pred, ct_ref).mean().item())

                if epoch >= start_64:
                    ct_pred, ct_ref = torch.clip(ct_pred_64, min=0, max=1), torch.clip(ct_ref_64, min=0, max=1)
                    SSIM_64.append(ssim(ct_pred, ct_ref).mean().item())
                    PSNR_64.append(psnr(ct_pred, ct_ref).mean().item())
                    LPIPS_64.append(lpips(ct_pred, ct_ref).mean().item())
                    L1_64.append(F.l1_loss(ct_pred, ct_ref).mean().item())
                    L2_64.append(F.mse_loss(ct_pred, ct_ref).mean().item())

                if epoch >= start_128:
                    ct_pred, ct_ref = torch.clip(ct_pred_128, min=0, max=1), torch.clip(ct_ref_128, min=0, max=1)
                    SSIM_128.append(ssim(ct_pred, ct_ref).mean().item())
                    PSNR_128.append(psnr(ct_pred, ct_ref).mean().item())
                    LPIPS_128.append(lpips(ct_pred, ct_ref).mean().item())
                    L1_128.append(F.l1_loss(ct_pred, ct_ref).mean().item())
                    L2_128.append(F.mse_loss(ct_pred, ct_ref).mean().item())

                if epoch >= start_256:
                    ct_pred, ct_ref = torch.clip(ct_pred_256, min=0, max=1), torch.clip(ct_ref_256, min=0, max=1)
                    SSIM_256.append(ssim(ct_pred, ct_ref).mean().item())
                    PSNR_256.append(psnr(ct_pred, ct_ref).mean().item())
                    LPIPS_256.append(lpips(ct_pred, ct_ref).mean().item())
                    L1_256.append(F.l1_loss(ct_pred, ct_ref).mean().item())
                    L2_256.append(F.mse_loss(ct_pred, ct_ref).mean().item())



        end_time = time.time()
        print(f"executing time: {end_time - start}s")
        ssim_mean = np.mean(SSIM_32)
        psnr_mean = np.mean(PSNR_32)
        lpips_mean = np.mean(LPIPS_32)
        l1_mean = np.mean(L1_32)
        l2_mean = np.mean(L2_32)
        text = f"epoch: {epoch}, ssim mean: {round(ssim_mean,6)}; psnr mean: {round(psnr_mean,6)}; lpips mean: {round(lpips_mean,6)}, l1 mean: {round(l1_mean,6) }; l2 mean: {round(l2_mean,6)}"
        print(text)
        logging.info(text)


        ssim_mean = np.mean(SSIM_64)
        psnr_mean = np.mean(PSNR_64)
        lpips_mean = np.mean(LPIPS_64)
        l1_mean = np.mean(L1_64)
        l2_mean = np.mean(L2_64)
        text = f"epoch: {epoch}, ssim mean: {round(ssim_mean,6)}; psnr mean: {round(psnr_mean,6)}; lpips mean: {round(lpips_mean,6)}, l1 mean: {round(l1_mean,6) }; l2 mean: {round(l2_mean,6)}"
        print(text)
        logging.info(text)


        ssim_mean = np.mean(SSIM_128)
        psnr_mean = np.mean(PSNR_128)
        lpips_mean = np.mean(LPIPS_128)
        l1_mean = np.mean(L1_128)
        l2_mean = np.mean(L2_128)
        text = f"epoch: {epoch}, ssim mean: {round(ssim_mean,6)}; psnr mean: {round(psnr_mean,6)}; lpips mean: {round(lpips_mean,6)}, l1 mean: {round(l1_mean,6) }; l2 mean: {round(l2_mean,6)}"
        print(text)
        logging.info(text)


        ssim_mean, ssim_std = np.mean(SSIM_256), np.std(SSIM_256)
        psnr_mean, psnr_std = np.mean(PSNR_256), np.std(PSNR_256)
        lpips_mean, lpips_std = np.mean(LPIPS_256), np.std(LPIPS_256)
        l1_mean, l1_std = np.mean(L1_256), np.std(L2_256)
        l2_mean, l2_std = np.mean(L2_256), np.std(L2_256)

        text = f"epoch: {epoch}, ssim mean: {round(ssim_mean,6)}, ssim std: {round(ssim_std,6)}; psnr mean: {round(psnr_mean,6)}, psnr std: {round(psnr_std,6)}; lpips mean: {round(lpips_mean,6)}, lpips std: {round(lpips_std,6)}, l1 mean: {round(l1_mean,6) }, l1 std: {round(l1_std,6)}; l2 mean: {round(l2_mean,6)}, l2 std: {round(l2_std,6)}"
        print(text)
        logging.info(text)

            

        if save_flag and epoch >= (start_256+10): 
            VAL_LOSSES_CT_256.append(ssim_mean)
            if ssim_mean == np.max(VAL_LOSSES_CT_256) and ('CT_256' in EMA):
                torch.save({'model_32': test_model_32.state_dict(), 'model_64': test_model_64.state_dict(), 'model_128': test_model_128.state_dict(), 'model_256': test_model_256.state_dict()}, model_save_path_256.format(str(epoch)+'-ct-best'))
    

    test_model_32.train()
    test_model_64.train()
    test_model_128.train()
    test_model_256.train()






def train_ct_256(modules, train_loader):
    optimizer_32 = torch.optim.AdamW(params= MODEL['CT_32'].parameters() , lr=learning_rate)
    optimizer_64 = torch.optim.AdamW(params= MODEL['CT_64'].parameters(), lr=learning_rate)
    optimizer_128 = torch.optim.AdamW(params= MODEL['CT_128'].parameters(), lr=learning_rate)
    optimizer_256 = torch.optim.AdamW(params= MODEL['CT_256'].parameters(), lr=learning_rate)

    scheduler_32 = torch.optim.lr_scheduler.MultiStepLR(optimizer_32, milestones=[end_32-15,end_32-5], gamma=0.25)
    scheduler_64 = torch.optim.lr_scheduler.MultiStepLR(optimizer_64, milestones=[end_64-15,end_64-5], gamma=0.25)
    scheduler_128 = torch.optim.lr_scheduler.MultiStepLR(optimizer_128, milestones=[end_128-15,end_128-5], gamma=0.25)
    scheduler_256 = torch.optim.lr_scheduler.MultiStepLR(optimizer_256, milestones=[end_256-15,end_256-5], gamma=0.25)

    epoch_start = start_32
    epoch_end = end_256
    for epoch in range(epoch_start, epoch_end):
        MODEL['CT_32'].train()
        MODEL['CT_64'].train()
        MODEL['CT_128'].train()
        MODEL['CT_256'].train()
        epoch_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=120)
        progress_bar.set_description(f"Epoch {epoch}")

        if epoch == (start_32+5): EMA['CT_32'] = ModelEMA(MODEL['CT_32'], decay=ema_rate)
        if epoch == (start_64+5): EMA['CT_64'] = ModelEMA(MODEL['CT_64'], decay=ema_rate)
        if epoch == (start_128+5): EMA['CT_128'] = ModelEMA(MODEL['CT_128'], decay=ema_rate)
        if epoch == (start_256+5): EMA['CT_256'] = ModelEMA(MODEL['CT_256'], decay=ema_rate)

        for step, batch in progress_bar:
            with torch.no_grad():
                ct_full = pre_process(batch, val_flag=False)
                modules.pre_cal(ct_full)
                modules.predict_sino_32()

            for i in range(math.ceil(ct_full.shape[0]/mini_batch_size_32)):
                RANGE=[i*mini_batch_size_32, i*mini_batch_size_32+(min(ct_full.shape[0]-i*mini_batch_size_32, mini_batch_size_32))]
                _, ref = modules.predict_ct_32(MODEL['CT_32'], RANGE)
                LOSS_32 = modules.loss_cal_ct_32(ref)

            if epoch >= start_64:
                with torch.no_grad(): modules.predict_sino_64()
                with contextlib.nullcontext() if epoch <= end_64 else torch.no_grad():
                    for i in range(math.ceil(ct_full.shape[0]/mini_batch_size_64)):
                        RANGE=[i*mini_batch_size_64, i*mini_batch_size_64+(min(ct_full.shape[0]-i*mini_batch_size_64, mini_batch_size_64))]
                        _, ref = modules.predict_ct_64(MODEL['CT_64'], RANGE)
                        LOSS_64 = modules.loss_cal_ct_64(ref)

            if epoch >= start_128:
                with torch.no_grad(): modules.predict_sino_128()
                with contextlib.nullcontext() if epoch <= end_128 else torch.no_grad():
                    for i in range(math.ceil(ct_full.shape[0]/mini_batch_size_128)):
                        RANGE=[i*mini_batch_size_128, i*mini_batch_size_128+(min(ct_full.shape[0]-i*mini_batch_size_128, mini_batch_size_128))]
                        _, ref = modules.predict_ct_128(MODEL['CT_128'], RANGE)
                        LOSS_128 = modules.loss_cal_ct_128(ref)

            if epoch >= start_256:
                with torch.no_grad(): modules.predict_sino_256()
                with contextlib.nullcontext() if epoch <= end_256 else torch.no_grad():
                    for i in range(math.ceil(ct_full.shape[0]/mini_batch_size_256)):
                        RANGE=[i*mini_batch_size_256, i*mini_batch_size_256+(min(ct_full.shape[0]-i*mini_batch_size_256, mini_batch_size_256))]
                        _, ref = modules.predict_ct_256(MODEL['CT_256'], RANGE)
                        LOSS_256 = modules.loss_cal_ct_256(ref)

            loss_32, loss_64, loss_128, loss_256 = 0, 0, 0, 0

            for j in range(len(LOSS_32)): loss_32 += LOSS_32[j]
            if epoch >= start_64: 
                for j in range(len(LOSS_64)): loss_64 += LOSS_64[j]
            if epoch >= start_128: 
                for j in range(len(LOSS_128)): loss_128 += LOSS_128[j]
            if epoch >= start_256: 
                for j in range(len(LOSS_256)): loss_256 += LOSS_256[j]
                
            if epoch <= end_32:
                optimizer_32.zero_grad()
                loss_32.backward()
                optimizer_32.step()
                if 'CT_32' in EMA: EMA['CT_32'].update(MODEL['CT_32'])

            if epoch >= start_64 and epoch <= end_64: 
                optimizer_64.zero_grad()
                loss_64.backward()
                optimizer_64.step()
                if 'CT_64' in EMA: EMA['CT_64'].update(MODEL['CT_64'])

            if epoch >= start_128 and epoch <= end_128: 
                optimizer_128.zero_grad()
                loss_128.backward()
                optimizer_128.step()
                if 'CT_128' in EMA: EMA['CT_128'].update(MODEL['CT_128'])

            if epoch >= start_256 and epoch <= end_256: 
                optimizer_256.zero_grad()
                loss_256.backward()
                optimizer_256.step()
                if 'CT_256' in EMA: EMA['CT_256'].update(MODEL['CT_256'])


            if epoch < start_64:
                epoch_loss = epoch_loss + loss_32.item()
            elif epoch < start_128:
                epoch_loss = epoch_loss + loss_64.item()
            elif epoch < start_256:
                epoch_loss = epoch_loss + loss_128.item()
            else:
                epoch_loss = epoch_loss + loss_256.item()
            progress_bar.set_postfix({"all": epoch_loss / (step*batch_size//mini_batch_size_256 + (i+1))}) 

        scheduler_32.step()
        scheduler_64.step()
        scheduler_128.step()
        scheduler_256.step()

        test_ct_256(modules, val_loader, EMA['CT_32'].get() if ('CT_32' in EMA) else MODEL['CT_32'], EMA['CT_64'].get() if ('CT_64' in EMA) else MODEL['CT_64'], EMA['CT_128'].get() if ('CT_128' in EMA) else MODEL['CT_128'], EMA['CT_256'].get() if ('CT_256' in EMA) else MODEL['CT_256'], save_flag=True, epoch=epoch)







modules = Fusion_Modules()

# model_path = '/DIM/89-ct-best.pth'
# state_dict = torch.load(model_path)
# MODEL['CT_32'].load_state_dict(state_dict['model_32'])
# MODEL['CT_64'].load_state_dict(state_dict['model_64'])
# MODEL['CT_128'].load_state_dict(state_dict['model_128'])
# MODEL['CT_256'].load_state_dict(state_dict['model_256'])
# test_ct_256(modules, val_loader, MODEL['CT_32'], MODEL['CT_64'], MODEL['CT_128'], MODEL['CT_256'], save_flag=False, epoch='val_set') 


train_ct_256(modules, train_loader)
