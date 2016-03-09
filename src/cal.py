'''
module holding all case-irrelevant calculations
'''
from PIL import Image
from scipy import optimize
import numpy as np
def scipy_optimize(name):
    def fmin(f, x):
        res = optimize.minimize(fun=f, x0=x, method=name)
        return res.x, res.fun
    return fmin

def scipy_optimize_jac(name):
    delta = 0.005        
    def fmin(f, x):
        res = optimize.minimize(fun=f, x0=x, method=name,
                    jac=_get_jac(f, delta, x))
        return res.x, res.fun
    return fmin

def _get_jac(func, delta, x0):
    # a gradient approximated jacobian computation
    # let func be the energy function and delta as the uniform delta for gradient
    len_x = len(x0)
    def jac(x):
        fx = func(x)
        grad = np.zeros(len_x)        
        for i in range(len_x):
            x_t = np.zeros(len_x)
            x_t[i] = delta
            fx_t = func(x+x_t)
            grad[i] = fx_t - fx
        return grad / delta
    return jac

def _sq_diff(a, b):
    # calculate the square difference of two equal-shaped numpy array
    return ((a-b)**2).sum()

_X, _Y = None, None

def _init_X_Y(width, height):
    # will be called by renderer when the initialization is done
    global _X, _Y
    _X = np.arange(width).reshape(1, width)
    _Y = np.arange(height).reshape(height, 1)
        
def _get_sec_moments(image):
    # image should be a gray scale Image object
    img = 1 - np.array(image.getdata()) / 128 # turn white to 0 and black to 1
    # using 128 in case of gray
    img = img.astype(np.int8)
    width, height = image.size
    img = img.reshape(height, width)
    M_00 = float(img.sum())   
#     if _X == None or _Y == None:
#         _init_X_Y(width, height)     
    M_10 = (_X * img).sum()
    M_01 = (img * _Y).sum()
    m_10 = M_10 / M_00 if M_00 else 0
    m_01 = M_01 / M_00 if M_00 else 0
    X_offset = _X - m_10
    Y_offset = _Y - m_01
    m_20 = ((X_offset**2)*img).sum() / M_00 if M_00 else 0
    m_02 = (img*(Y_offset**2)).sum() / M_00 if M_00 else 0
    m_11 = (X_offset*img*Y_offset).sum() / M_00 if M_00 else 0
    return np.array([m_20, m_11, m_02])    
    
def get_fst_moments(image):
    # image should be a gray scale Image object
    img = 1 - np.array(image.getdata()) / 128 # turn white to 0 and black to 1
    # using 128 in case of gray
    img = img.astype(np.int8)
    width, height = image.size
    img = img.reshape(height, width)
    M_00 = float(img.sum())        
    if _X == None or _Y == None:
        _init_X_Y(width, height)
    M_10 = (_X * img).sum()
    M_01 = (img * _Y).sum()
    m_10 = M_10 / M_00 if M_00 else 0
    m_01 = M_01 / M_00 if M_00 else 0
    return np.array([m_10, m_01])

from PIL import ImageMath as imath    
def _xor_closure(target):
    def _get_xor(image):
        xor_img = imath.eval("a^b", a=image, b=target)
        return sum(xor_img.getdata()) / (640*480)
    return _get_xor

def _sq_diff_closure(func):
    def sub_closure(target):
        res_t = func(target)
        def sqdiff(image):
            return _sq_diff(res_t, func(image))
        return sqdiff
    return sub_closure
