import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os, shutil
from distutils.dir_util import copy_tree


import numpy as np
import tensorflow as tf
import os, shutil
from keras import backend as K
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Add, BatchNormalization, Concatenate, Conv2DTranspose, ZeroPadding2D, Cropping2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad
from tensorflow.keras.callbacks import History, ModelCheckpoint
import matplotlib.pyplot as plt
import glob
import cv2
from PIL import Image, ImageFilter
from scipy.ndimage.filters import gaussian_filter
K.set_image_data_format("channels_last")




def make_sketches(path_in, path_sketch):
    
    """
    This takes one of the real images and creates the "blurred" or modified images
    path in is where the input images are, path sketch is the place where the modified ones will be stored.
    """
    
    os.chdir(path_in)
    file_list = glob.glob("*")
    file_list.sort()
#   os.chdir(path_sketch)

    for file in file_list:
        a = Image.open(file)
        a = a.filter(ImageFilter.GaussianBlur(radius = 2))
        a = a.filter(ImageFilter.GaussianBlur(radius = 4))
        a = a.filter(ImageFilter.GaussianBlur(radius = 10))
        a = ImageOps.posterize(a, 4)
        b = np.asarray(a).copy() # Returns a view, not the array!! Need a copy to assign and play with it.
        b[b[:, :, 1] < 100] = 0
        im = Image.fromarray(np.uint8(b))
        os.chdir(path_sketch)
        im.save(file)
        os.chdir(path_in)
#         print(b.shape)
        
        
    
    return None
