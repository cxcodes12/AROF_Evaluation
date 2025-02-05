
import numpy as np
import matplotlib.pyplot as plt
from skimage import color, io
from scipy import signal
from skimage.metrics import structural_similarity as ssim


def calcul_SNR(im_f, im_ref):
    signal_power = np.sum(im_ref**2)
    noise_power = np.sum((im_ref - im_f)**2)
    SNR = 10*np.log10(signal_power/noise_power)
    return SNR

def calcul_PSNR(im_f, im_ref):
    mse = np.mean((im_ref - im_f)**2)
    if mse==0:
        return float('inf')
    PSNR = 10*np.log10(1/mse) # 1 = val maxima a pixelilor
    return PSNR

def calcul_MAE(im_f, im_ref):
    MAE = np.mean(np.abs(im_f - im_ref))
    return MAE

def calcul_SSIM(im_f, im_ref):
    ssim_values = []
    for i in range(3):  # canalele R, G, B
        ssim_value = ssim(
            im_ref[:, :, i],
            im_f[:, :, i],
            data_range=im_ref[:, :, i].max() - im_ref[:, :, i].min()  
        )
        ssim_values.append(ssim_value)
    return sum(ssim_values) / len(ssim_values)  # media SSIM pe toate canalele

    