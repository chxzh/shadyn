'''
module holding all case-irrelevant calculations
'''
from PIL import Image
from scipy import optimize
import numpy as np
from cgkit.cgtypes import *
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

def init_X_Y(width, height):
    # will be called by renderer when the initialization is done
    global _X, _Y
    _X = np.arange(width).reshape(1, width)
    _Y = np.arange(height).reshape(height, 1)
        
def get_sec_moments(image, path):
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
    
def get_fst_moments(image, path):
    # image should be a gray scale Image object
    img = 1 - np.array(image.getdata()) / 128 # turn white to 0 and black to 1
    # using 128 in case of gray
    img = img.astype(np.int8)
    width, height = image.size
    img = img.reshape(height, width)
    M_00 = float(img.sum())        
    if _X == None or _Y == None:
        init_X_Y(width, height)
    M_10 = (_X * img).sum()
    M_01 = (img * _Y).sum()
    m_10 = M_10 / M_00 if M_00 else 0
    m_01 = M_01 / M_00 if M_00 else 0
    return np.array([m_10, m_01])

from PIL import ImageMath as imath    
def xor_closure(target, path):
    '''
    target shall be a single-band Image instance
    '''
    target = imath.eval("x>>7", x=target)
    def _get_xor(image):
        xor_img = imath.eval("(a>>7)^b", a=image, b=target)
        return sum(xor_img.getdata())
    return _get_xor

import os.path
def distance_field_closure(target, path, radius=1):    
    root, ext = os.path.splitext(path)
    df_cache_path = root + '_df' + ".png"
    if os.path.exists(df_cache_path):        
        target = Image.open(df_cache_path)
        target = target.convert('L')
    else:
        target = get_distance_field(target, radius)        
        target.save(df_cache_path)
    def _get_xor(image):
        xor_img = imath.eval("t*(1-(x>>7))", t=target, x=image)
        return sum(xor_img.getdata())
    return _get_xor

import cv2
def get_distance_field(target, radius=1):
    kernel = np.ones((1+radius*2,1+radius*2), np.uint8)
    counter = 1
    original = np.array(target)
    res = original.copy()
    def expand(a, b, value):
        if a <= value or a == b:
            return a
        else:
            return value
    expand = lambda a, b, val: a if a <= val or a == b else val
    expand = np.vectorize(expand)
    expanded = original.copy()
    for i in xrange(1, 256):
        expanded = cv2.erode(expanded, kernel)
        res = expand(res, expanded, i)
    res = Image.fromarray(res)
    return res

def test_erosion():
    from PIL import Image
    target = Image.open("../img/target_spade.png").convert('L')
#     image = Image.open("../img/target_forward.png").convert('L')
#     f = uncovered_closure(target)
#     res = f(image)
    get_distance_field(target, 1).show()
    
def uncovered_closure(target, path):
    def _uncover(image): 
        res = imath.eval("((~a)>>7)&(b>>7)", a=target, b=image)
        return sum(res.getdata())
    return _uncover

def sq_diff_closure(func):
    def sub_closure(target, path):
        res_t = func(target, path)
        def sqdiff(image):
            return _sq_diff(res_t, func(image, path))
        return sqdiff
    return sub_closure

def shadow_proj_mat(plane_normal, point_sample, light_pos):
    if type(plane_normal) is vec3:
        plane_normal = plane_normal.normalize()
    elif type(plane_normal) is np.array:
        plane_normal = plane_normal / np.linalg.norm(plane_normal)
    else:
        raise TypeError("What the hell did you put in here as a normal??")
    n_t = np.array(plane_normal).reshape((1, 3))
    L = np.array(light_pos).reshape((3, 1))
    D = -plane_normal * point_sample
    ntL = np.dot(n_t, L)
    shad_mat = np.identity(4, float)
    shad_mat[0:3, 0:3] = L.dot(n_t) - (D + ntL) * np.identity(3)
    shad_mat[0:3, 3:4] = (D + ntL) * L - L * ntL
    shad_mat[3:4, 0:3] = n_t
    shad_mat[3:4, 3:4] = -ntL
    return mat4(shad_mat.astype(np.float32).T.tolist())

def _main():
    test_erosion()

if __name__ == '__main__':
    _main()