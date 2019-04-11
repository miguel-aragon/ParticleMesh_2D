import numpy as np
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import skcuda.fft as cu_fft
from skimage import color
import matplotlib.pyplot as plt

#--------------------------------------------------------------
#
#--------------------------------------------------------------
def tvscale(img, wsize=-1,scale=0, invert=False, _plot_size=(6,6)):

    plt.rcParams['figure.figsize'] = [_plot_size[0], _plot_size[1]]
    
    #--- Select image scaling
    if scale==0:
        im = img
    if scale==1:
        im = (img-np.min(img)) / (np.max(img)-np.min(img)).astype(float)

    if invert==True:
        imgplot = plt.imshow(img,interpolation="none", origin='lower',cmap='gray')
    else:
        imgplot = plt.imshow(img,interpolation="none",cmap='gray')
    plt.axis('off')
    plt.show()
  
#--------------------------------------------------------------
#  https://www.idtools.com.au/gpu-accelerated-fft-compatible-numpy/
#--------------------------------------------------------------
def fft2_gpu(x, fftshift=False):
    
    ''' This function produce an output that is 
    compatible with numpy.fft.fft2
    The input x is a 2D numpy array'''

    # Convert the input array to single precision float
    if x.dtype != 'float32':
        x = x.astype('float32')

    # Get the shape of the initial numpy array
    n1, n2 = x.shape
    
    # From numpy array to GPUarray
    xgpu = gpuarray.to_gpu(x)
    
    # Initialise output GPUarray 
    # For real to complex transformations, the fft function computes 
    # N/2+1 non-redundant coefficients of a length-N input signal.
    y = gpuarray.empty((n1,n2//2 + 1), np.complex64)
    
    # Forward FFT
    plan_forward = cu_fft.Plan((n1, n2), np.float32, np.complex64)
    cu_fft.fft(xgpu, y, plan_forward)
    
    left = y.get()

    # To make the output array compatible with the numpy output
    # we need to stack horizontally the y.get() array and its flipped version
    # We must take care of handling even or odd sized array to get the correct 
    # size of the final array   
    if n2//2 == n2/2:
        right = np.roll(np.fliplr(np.flipud(y.get()))[:,1:-1],1,axis=0)
    else:
        right = np.roll(np.fliplr(np.flipud(y.get()))[:,:-1],1,axis=0) 
    
    # Get a numpy array back compatible with np.fft
    if fftshift is False:
        yout = np.hstack((left,right))
    else:
        yout = np.fft.fftshift(np.hstack((left,right)))

    return yout.astype('complex128')

#--------------------------------------------------------------
#  https://www.idtools.com.au/gpu-accelerated-fft-compatible-numpy/
#--------------------------------------------------------------
def ifft2_gpu(y, fftshift=False):

    ''' This function produce an output that is 
    compatible with numpy.fft.ifft2
    The input y is a 2D complex numpy array'''
 
    # Get the shape of the initial numpy array
    n1, n2 = y.shape
    
    # From numpy array to GPUarray. Take only the first n2/2+1 non redundant FFT coefficients
    if fftshift is False:
        y2 = np.asarray(y[:,0:n2//2 + 1], np.complex64)
    else:
        y2 = np.asarray(np.fft.ifftshift(y)[:,:n2//2+1], np.complex64)
    ygpu = gpuarray.to_gpu(y2) 
     
    # Initialise empty output GPUarray 
    x = gpuarray.empty((n1,n2), np.float32)
    
    # Inverse FFT
    plan_backward = cu_fft.Plan((n1, n2), np.complex64, np.float32)
    cu_fft.ifft(ygpu, x, plan_backward)
    
    # Must divide by the total number of pixels in the image to get the normalisation right
    xout = x.get()/n1/n2
    
    return xout

im = plt.imread('galaxy.jpg').astype('uint16')
img = color.rgb2gray(im)

fft1 = np.fft.fftshift(np.fft.fft2(img))
fft2 = fft2_gpu(img, fftshift=True)


img1 = np.real(np.fft.ifft2(np.fft.ifftshift(fft1)))
img2 = ifft2_gpu(fft2, fftshift=True)









