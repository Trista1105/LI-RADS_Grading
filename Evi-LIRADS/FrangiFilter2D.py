import numpy as np
import cv2
import math
from scipy import signal


def Hessian2D0(I,Sigma):
    #求输入图像的Hessian矩阵
    #实际上中间包含了高斯滤波，因为二阶导数对噪声非常敏感
    #输入：    
    #   I : 单通道double类型图像
    #   Sigma : 高斯滤波卷积核的尺度  
    #    
    # 输出 : Dxx, Dxy, Dyy: 图像的二阶导数 
    if Sigma<1:
        print("error: Sigma<1")
        return -1
    I=np.array(I,dtype=float)
    Sigma=np.array(Sigma,dtype=float)
    S_round=np.round(3*Sigma)

    [X,Y]= np.mgrid[-S_round:S_round+1,-S_round:S_round+1]

    #构建卷积核：高斯函数的二阶导数
    DGaussxx = 1/(2*math.pi*pow(Sigma,4)) * (X**2/pow(Sigma,2) - 1) * np.exp(-(X**2 + Y**2)/(2*pow(Sigma,2)))
    DGaussxy = 1/(2*math.pi*pow(Sigma,6)) * (X*Y) * np.exp(-(X**2 + Y**2)/(2*pow(Sigma,2)))   
    DGaussyy = 1/(2*math.pi*pow(Sigma,4)) * (Y**2/pow(Sigma,2) - 1) * np.exp(-(X**2 + Y**2)/(2*pow(Sigma,2)))
  
    Dxx = signal.convolve2d(I,DGaussxx,boundary='fill',mode='same',fillvalue=0)
    Dxy = signal.convolve2d(I,DGaussxy,boundary='fill',mode='same',fillvalue=0)
    Dyy = signal.convolve2d(I,DGaussyy,boundary='fill',mode='same',fillvalue=0)

    return Dxx,Dxy,Dyy


def eig2image0(Dxx,Dxy,Dyy):
    # This function eig2image calculates the eigen values from the
    # hessian matrix, sorted by abs value. And gives the direction
    # of the ridge (eigenvector smallest eigenvalue) .
    # input:Dxx,Dxy,Dyy图像的二阶导数
    # output:Lambda1,Lambda2,Ix,Iy
    #Compute the eigenvectors of J, v1 and v2
    Dxx=np.array(Dxx,dtype=float)
    Dyy=np.array(Dyy,dtype=float)
    Dxy=np.array(Dxy,dtype=float)
    if (len(Dxx.shape)!=2):
        print("len(Dxx.shape)!=2,Dxx不是二维数组！")
        return 0

    tmp = np.sqrt( (Dxx - Dyy)**2 + 4*Dxy**2)

    v2x = 2*Dxy
    v2y = Dyy - Dxx + tmp

    mag = np.sqrt(v2x**2 + v2y**2)
    i=np.array(mag!=0)

    v2x[i==True] = v2x[i==True]/mag[i==True]
    v2y[i==True] = v2y[i==True]/mag[i==True]

    v1x = -v2y 
    v1y = v2x

    mu1 = 0.5*(Dxx + Dyy + tmp)
    mu2 = 0.5*(Dxx + Dyy - tmp)

    check=abs(mu1)>abs(mu2)
            
    Lambda1=mu1.copy()
    Lambda1[check==True] = mu2[check==True]
    Lambda2=mu2
    Lambda2[check==True] = mu1[check==True]
    
    Ix=v1x
    Ix[check==True] = v2x[check==True]
    Iy=v1y
    Iy[check==True] = v2y[check==True]
    
    return Lambda1,Lambda2,Ix,Iy, mu1, mu2, v1x, v1y, v2x,v2y


def FrangiFilter2D0(I, frangiBetaTwo):

    I=np.array(I,dtype=float)
    defaultoptions = {'FrangiScaleRange':(1,10), 'FrangiScaleRatio':2, 'FrangiBetaOne':0.5, 'FrangiBetaTwo':15, 'verbose':True,'BlackWhite':True};
    #options=defaultoptions
    options = {'FrangiScaleRange': (1, 2), 'FrangiScaleRatio': 1, 'FrangiBetaOne':0.5, 'FrangiBetaTwo': frangiBetaTwo,
               'verbose': False, 'BlackWhite': False};  # 'FrangiBetaTwo': 0.07/0.05/0.03, 0.04; 'FrangiBetaOne': 0.5  #LIRADS: 'BlackWhite': False


    sigmas=np.arange(options['FrangiScaleRange'][0],options['FrangiScaleRange'][1],options['FrangiScaleRatio'])
    sigmas.sort()#升序

    beta  = 2*pow(options['FrangiBetaOne'],2)  
    c     = 2*pow(options['FrangiBetaTwo'],2)

    #存储滤波后的图像
    shape=(I.shape[0],I.shape[1],len(sigmas))
    ALLfiltered=np.zeros(shape) 
    ALLangles  =np.zeros(shape) 

    #Frangi filter for all sigmas 
    Rb=0
    S2=0

    for i in range(len(sigmas)):
        #Show progress
        if(options['verbose']):
            print('Current Frangi Filter Sigma: ',sigmas[i])
        
        #Make 2D hessian
        [Dxx,Dxy,Dyy] = Hessian2D(I,sigmas[i])

        #Correct for scale 
        Dxx = pow(sigmas[i],2)*Dxx  
        Dxy = pow(sigmas[i],2)*Dxy  
        Dyy = pow(sigmas[i],2)*Dyy

        c1 = []
        for j in range (0, I.shape[0]-1):
            for k in range(0,I.shape[1]-1):
                H = [[Dxx[j, k], Dxy[j, k]], [Dxy[j, k], Dyy[j,k]]]
                c2 = np.linalg.norm(H, float('inf')) #1,列和范数。2，最大特征值，谱范数。 inf, 行和范数
                c1.append(c2)


        # c = 2 * pow(0.5*max(c1), 2)

        # print('c is: ', 0.5*max(c1))

        #Calculate (abs sorted) eigenvalues and vectors  
        [Lambda2,Lambda1,Ix,Iy, mu1,mu2,v1x,v1y,v2x, v2y]=eig2image(Dxx,Dxy,Dyy)

        #Compute the direction of the minor eigenvector  
        angles = np.arctan2(Ix,Iy)  #较小特征值对应的方向，血管方向

        #Compute some similarity measures  
        Lambda1[Lambda1==0] = np.spacing(1)

        Rb = (Lambda2/Lambda1)**2
        S2 = Lambda1**2 + Lambda2**2
        
        #Compute the output image
        Ifiltered = np.exp(-Rb/beta) * (np.ones(I.shape)-np.exp(-S2/c))
         
        #see pp. 45  
        if(options['BlackWhite']): 
            Ifiltered[Lambda1<0]=0
        else:
            Ifiltered[Lambda1>0]=0
        
        #store the results in 3D matrices  
        ALLfiltered[:,:,i] = Ifiltered 
        ALLangles[:,:,i] = angles

        # Return for every pixel the value of the scale(sigma) with the maximum   
        # output pixel value  
        if len(sigmas) > 1:
            outIm=ALLfiltered.max(2)
        else:
            outIm = ALLfiltered
        # else:
        #     outIm = (outIm.transpose()).reshape(I.shape)
            
    return Ifiltered, angles, mu1, mu2, v1x, v1y, v2x, v2y,S2





def Hessian2D(I, Sigma):
    """Compute the Hessian matrix using the original convolution method."""
    I = np.array(I, dtype=float)
    S_round = np.round(3 * Sigma)

    [X, Y] = np.mgrid[-S_round:S_round + 1, -S_round:S_round + 1]

    DGaussxx = 1 / (2 * np.pi * Sigma ** 4) * (X ** 2 / Sigma ** 2 - 1) * np.exp(-(X ** 2 + Y ** 2) / (2 * Sigma ** 2))
    DGaussxy = 1 / (2 * np.pi * Sigma ** 6) * (X * Y) * np.exp(-(X ** 2 + Y ** 2) / (2 * Sigma ** 2))
    DGaussyy = 1 / (2 * np.pi * Sigma ** 4) * (Y ** 2 / Sigma ** 2 - 1) * np.exp(-(X ** 2 + Y ** 2) / (2 * Sigma ** 2))

    # Reverting to the original convolution method
    Dxx = signal.convolve2d(I, DGaussxx, boundary='fill', mode='same', fillvalue=0)
    Dxy = signal.convolve2d(I, DGaussxy, boundary='fill', mode='same', fillvalue=0)
    Dyy = signal.convolve2d(I, DGaussyy, boundary='fill', mode='same', fillvalue=0)

    return Dxx, Dxy, Dyy


def eig2image(Dxx, Dxy, Dyy):
    """Compute the eigenvalues and eigenvectors of the Hessian matrix."""
    Dxx = np.array(Dxx, dtype=float)
    Dyy = np.array(Dyy, dtype=float)
    Dxy = np.array(Dxy, dtype=float)

    tmp = np.sqrt((Dxx - Dyy) ** 2 + 4 * Dxy ** 2)

    v2x = 2 * Dxy
    v2y = Dyy - Dxx + tmp

    mag = np.sqrt(v2x ** 2 + v2y ** 2)
    i = np.array(mag != 0)

    v2x[i == True] = v2x[i == True] / mag[i == True]
    v2y[i == True] = v2y[i == True] / mag[i == True]

    v1x = -v2y
    v1y = v2x

    mu1 = 0.5 * (Dxx + Dyy + tmp)
    mu2 = 0.5 * (Dxx + Dyy - tmp)

    check = np.abs(mu1) > np.abs(mu2)

    Lambda1 = mu1.copy()
    Lambda1[check == True] = mu2[check == True]
    Lambda2 = mu2
    Lambda2[check == True] = mu1[check == True]

    Ix = v1x
    Ix[check == True] = v2x[check == True]
    Iy = v1y
    Iy[check == True] = v2y[check == True]

    return Lambda1, Lambda2, Ix, Iy, mu1, mu2, v1x, v1y, v2x, v2y


def FrangiFilter2D(I, frangiBetaTwo):
    I = np.array(I, dtype=float)
    options = {'FrangiScaleRange': (1, 2), 'FrangiScaleRatio': 1, 'FrangiBetaOne': 0.5, 'FrangiBetaTwo': frangiBetaTwo,
               'verbose': False, 'BlackWhite': False}

    sigmas = np.arange(options['FrangiScaleRange'][0], options['FrangiScaleRange'][1], options['FrangiScaleRatio'])
    sigmas.sort()

    beta = 2 * options['FrangiBetaOne'] ** 2
    c = 2 * options['FrangiBetaTwo'] ** 2

    ALLfiltered = np.zeros((I.shape[0], I.shape[1], len(sigmas)))
    ALLangles = np.zeros((I.shape[0], I.shape[1], len(sigmas)))

    for i, sigma in enumerate(sigmas):
        if options['verbose']:
            print(f'Current Frangi Filter Sigma: {sigma}')

        Dxx, Dxy, Dyy = Hessian2D(I, sigma)

        Dxx = sigma ** 2 * Dxx
        Dxy = sigma ** 2 * Dxy
        Dyy = sigma ** 2 * Dyy

        Lambda2, Lambda1, Ix, Iy, _, _, _, _, _, _ = eig2image(Dxx, Dxy, Dyy)

        angles = np.arctan2(Ix, Iy)

        Lambda1[Lambda1 == 0] = np.spacing(1)

        Rb = (Lambda2 / Lambda1) ** 2
        S2 = Lambda1 ** 2 + Lambda2 ** 2

        Ifiltered = np.exp(-Rb / beta) * (1 - np.exp(-S2 / c))

        if options['BlackWhite']:
            Ifiltered[Lambda1 < 0] = 0
        else:
            Ifiltered[Lambda1 > 0] = 0

        ALLfiltered[:, :, i] = Ifiltered
        ALLangles[:, :, i] = angles

    if len(sigmas) > 1:
        outIm = ALLfiltered.max(2)
    else:
        outIm = ALLfiltered[:, :, 0]

    return outIm, ALLangles[:, :, i]



if __name__ == "__main__":
    imagename='test.tif'
    image=cv2.imread(imagename,0)
    blood = cv2.normalize(image.astype('double'), None, 0.0, 1.0, cv2.NORM_MINMAX) # Convert to normalized floating point
    outIm=FrangiFilter2D(blood)
    img=outIm*(10000)

    cv2.imshow('img',img)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
