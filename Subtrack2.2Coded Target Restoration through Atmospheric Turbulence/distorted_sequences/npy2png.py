'''
Convert npy file to mat file
'''
import os
from pickletools import uint8
from traceback import FrameSummary
from types import GeneratorType 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

low_frames = '.\\distorted_sequences_npy\\low\\'
frames_dir = os.listdir(low_frames)
for frame_dir in frames_dir:
    data = np.load(os.path.join(low_frames,frame_dir))
    data_size = data.shape
    data_length = data_size[2]
    name = frame_dir[:-4]
    if not os.path.exists('./distorted_sequences_png/'+name+'/'):
        os.mkdir('./distorted_sequences_png/'+name+'/')
    for i in range(data_length):
        data1 = data[:,:,i]
        min_16bit = np.min(data1)
        max_16bit = np.max(data1)
        image_uint8 = np.array(((data1 - min_16bit) / (max_16bit - min_16bit)), dtype=float)
        cv2.imwrite('./distorted_sequences_png/' + name +'/'+str(i + 1)+'.png',image_uint8 * 255)
        # sio.savemat('./distorted_sequences_mat/'+name+'/'+str(i + 1)+'.mat', {'matrix': image_float})
