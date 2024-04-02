import os
from traceback import FrameSummary
from types import GeneratorType 
import cv2
import numpy as np


def image_enhance_gamma(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = 255 * np.power(img / 255, 2.5)
    img = np.around(img)
    img[img > 255] = 255
    out = img.astype(np.uint8)
    return out
imgs_dir = 'G:\\UG2+competation_submit\\subtrack2.2\\code\\method\step3_postprocess\\results\\'
imgs = os.listdir(imgs_dir)
for img in imgs:
    img_name = img[:-4]
    img_dir1 = os.path.join(imgs_dir,img)
    image = cv2.imread(img_dir1,0)
    image_gamma = image_enhance_gamma(image)
    img8bit_max = np.max(image_gamma)
    img8bit_min = np.min(image_gamma)
    image_16bit = np.array(np.rint(4096 * ((image_gamma - img8bit_min) / (img8bit_max - img8bit_min))), dtype=np.uint16)
    np.save('G:\\UG2+competation_submit\\subtrack2.2\\code\\method\\final_result_npy\\'+img_name+'_out.npy',image_16bit)