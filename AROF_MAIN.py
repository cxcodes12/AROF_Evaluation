
%clear
import numpy as np
import matplotlib.pyplot as plt
from skimage import color, io
from scipy import signal

plt.close('all')

from my_functions import add_impulsive_noise, add_gaussian_noise, AROF, AMF, median_filter, medie_aritmetica
from masuri_calitate import calcul_SNR, calcul_PSNR, calcul_MAE, calcul_SSIM

# citire imagine color
img = io.imread('lena.png')
img = img[:,:,0:3]/255
plt.figure(),plt.subplot(3,2, 1),plt.axis('off'),plt.imshow(img),plt.title('Original Image')

# adaugare zgomot impulsiv
DENSITATE_01 = 0.8

img_n = np.zeros((np.shape(img)))
img_n = np.copy(img)
img_n = add_impulsive_noise(img_n,DENSITATE_01)
plt.subplot(3,2, 2),plt.axis('off'), plt.imshow(img_n),plt.title('Impulsive noise')

# # adaugare zgomot gaussian
# img_n = np.zeros((np.shape(img)))
# img_n = np.copy(img)
# img_n = add_gaussian_noise(img_n, 150)
# plt.subplot(3,2, 2),plt.axis('off'),plt.imshow(img_n),plt.title('Gaussian noise')

#filtrare imagine cu AROF
img_arof = np.zeros(np.shape(img_n))
for p in range(0,3):
    plan = img_n[:,:,p]
    plan_filtrat = AROF(plan)
    img_arof[:,:,p] = plan_filtrat
    
plt.subplot(3,2, 3),plt.axis('off'),plt.imshow(img_arof),plt.title('AROF')


# calcul masuri calitate AROF
snr_arof = calcul_SNR(img_arof, img)
psnr_arof = calcul_PSNR(img_arof, img)
mae_arof = calcul_MAE(img_arof, img)
ssim_arof = calcul_SSIM(img_arof, img)

# #filtrare imagine cu AMF
img_amf = np.zeros(np.shape(img_n))
for p in range(0,3):
    plan = img_n[:,:,p]
    plan_filtrat = AMF(plan)
    img_amf[:,:,p] = plan_filtrat
    
plt.subplot(3,2, 4),plt.axis('off'),plt.imshow(img_amf),plt.title('Adaptive Median Filter')

# calcul masuri calitate AMF
snr_AMF = calcul_SNR(img_amf, img)
psnr_AMF = calcul_PSNR(img_amf, img)
mae_AMF = calcul_MAE(img_amf, img)
ssim_AMF = calcul_SSIM(img_amf, img)

#filtrare imagine cu filtru median 3x3
img_MF = np.zeros(np.shape(img_n))
for p in range(0,3):
    plan = img_n[:,:,p]
    plan_filtrat = median_filter(plan)
    img_MF[:,:,p] = plan_filtrat
    
plt.subplot(3,2, 5),plt.axis('off'),plt.imshow(img_MF),plt.title('Median Filter')

# calcul masuri calitate filtru median
snr_MF = calcul_SNR(img_MF, img)
psnr_MF = calcul_PSNR(img_MF, img)
mae_MF = calcul_MAE(img_MF, img)
ssim_MF = calcul_SSIM(img_MF, img)

#filtrare imagine cu filtru medie aritmetica 3x3
img_MA = np.zeros(np.shape(img_n))
for p in range(0,3):
    plan = img_n[:,:,p]
    plan_filtrat = medie_aritmetica(plan)
    img_MA[:,:,p]= plan_filtrat

plt.subplot(3,2, 6),plt.axis('off'),plt.subplots_adjust(wspace=0.1, hspace=0.1),plt.imshow(img_MA),plt.title('Arithmetic Mean Filter'),plt.show()

# calcul masuri calitate filtru medie aritmetica

snr_MA = calcul_SNR(img_MA, img)
psnr_MA = calcul_PSNR(img_MA, img)
mae_MA = calcul_MAE(img_MA, img)
ssim_MA = calcul_SSIM(img_MA, img)
















