import numpy as np
import matplotlib.pyplot as plt
from skimage import color, io
from scipy import signal

# functie de adaugare zgomot impulsiv peste o imagine RGB

def add_impulsive_noise(img, p):
    if (np.max(img)>1):
        img = img/255.0
    [M,N,P] = np.shape(img)
    
    img_zg = img.copy()
    nr_pix01 = int(p*M*N)
    for i in range(0,P):
        indici = np.random.choice(M*N, size=nr_pix01, replace=False)
        linii, col = np.unravel_index(indici, (M, N))
        
        img_zg[linii[0:nr_pix01//2], col[0:nr_pix01//2], i] = 1
        img_zg[linii[nr_pix01//2:], col[nr_pix01//2:], i] = 0
        
    return img_zg

# functie de adaugare zgomot gaussian peste o imagine RGB

def add_gaussian_noise(img, disp):
    # dispersia va fi data in intervalul 0-255
    disp = disp/255.0
    if (np.max(img)>1):
        img = img/255.0
    [M,N,P] = np.shape(img)
    img_zg = np.zeros((M,N,P))
    
    for i in range(0,P):
        zg = np.random.normal(0, disp, (M,N))
        img_zg[:,:,i] = img[:,:,i] + zg
    img_zg = np.clip(img_zg,a_min=0,a_max=1)
    
    return img_zg
        
# functie Adaptive Rank Order Filter pentru filtrarea unei imagini grayscale

def AROF(img):
    if (np.max(img)>1):
        img = img/255.0
    w = 1
    kernel = np.zeros((2*w+1,2*w+1))
    kernel[w,w]=1
    img_b = np.zeros(np.shape(img))
    img_b = signal.convolve2d(img,kernel,mode='full',boundary='symm')
    img_f = np.zeros(np.shape(img_b))
    [M,N] = np.shape(img_b)
        
    for i in range(w,M-w):
        for j in range(w,N-w):
            ok=0
            w=1
            while ok!=1:
                if img_b[i,j] != 0 and img_b[i,j] != 1:
                    img_f[i,j] = img_b[i,j]
                    ok=1
                else:
                    roi = img_b[i-w:i+w+1,j-w:j+w+1]
                    sorted_roi = np.sort(roi.flatten())
                    med = sorted_roi[np.size(sorted_roi)//2]
                    
                    if med != 0 and med != 1:
                        img_f[i,j] = med
                        ok=1
                    else:
                        poz = (sorted_roi != 0) & (sorted_roi != 1)
                        not_noisy = sorted_roi[poz]
                        
                        if np.sum(not_noisy)!=0:
                            diff = abs(img_b[i,j] - not_noisy)
                            img_f[i,j] = not_noisy[np.argmin(diff)]
                            ok=1
                        elif i+w<M-1 and j+w<N-1 and i-w>0 and j-w>0:
                            w=w+1
                            ok=0
                        else: ok=1
    img_f = img_f[1:-1,1:-1]
    return img_f

# functie Adaptive Median Filter

def AMF(img):
    if (np.max(img)>1):
        img = img/255.0
    w = 1
    kernel = np.zeros((2*w+1,2*w+1))
    kernel[w,w]=1
    img_b = np.zeros(np.shape(img))
    img_b = signal.convolve2d(img,kernel,mode='full',boundary='symm')
    img_f = np.zeros(np.shape(img_b))
    [M,N] = np.shape(img_b)
        
    for i in range(w,M-w):
        for j in range(w,N-w):
            ok=0
            w=1
            while ok!=1:
                if img_b[i,j] != 0 and img_b[i,j] != 1:
                    img_f[i,j] = img_b[i,j]
                    ok=1
                else:
                    roi = img_b[i-w:i+w+1,j-w:j+w+1]
                    sorted_roi = np.sort(roi.flatten())
                    med = sorted_roi[np.size(sorted_roi)//2]
                    
                    if med != 0 and med != 1:
                        img_f[i,j] = med
                        ok=1
                    elif i+w<M-1 and j+w<N-1 and i-w>0 and j-w>0:
                        w=w+1
                        ok=0
                    else: ok=1
                        
    img_f = img_f[1:-1,1:-1]
    return img_f

# functie filtru median - pentru o imagine grayscale

def median_filter(img):
    if np.max(img)>1:
        img = img/255.0
    
    w=1 #filtru 3x3
    kernel = np.zeros((2*w+1,2*w+1))
    kernel[w,w] = 1
    img_b = signal.convolve2d(img,kernel,mode='full',boundary='symm')
    img_f = np.zeros(np.shape(img_b))
    
    [M,N] = np.shape(img_b)
    for i in range(w,M-w):
        for j in range(w,N-w):
            roi = img_b[i-w:i+w+1, j-w:j+w+1]
            sorted_roi = np.sort(roi.flatten())
            img_f[i,j] = sorted_roi[np.size(sorted_roi)//2]
            
    img_f = img_f[1:-1,1:-1]
    return img_f
            
# functie filtru medie aritmetica - pentru o imagine grayscale

def medie_aritmetica(img):
    if (np.max(img)>1):
        img = img/255.0 
    
    w=1 #filtru 3x3
    kernel = np.zeros((2*w+1,2*w+1))
    kernel[w,w] = 1 
    img_b = signal.convolve2d(img,kernel,mode='full',boundary='symm')
    img_f = np.zeros(np.shape(img_b))
    
    [M,N] = np.shape(img_b)
    for i in range(w,M-w):
        for j in range(w,N-w):
            roi = img_b[i-w:i+w+1, j-w:j+w+1]
            img_f[i,j] = np.sum(roi)/np.size(roi)
    
    img_f = img_f[1:-1,1:-1]
    return img_f
            
































