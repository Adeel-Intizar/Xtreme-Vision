import cv2
import math
import time
import numpy as np
from . import util
import tensorflow as tf
from .config_reader import config_reader
from scipy.ndimage.filters import gaussian_filter
from tensorflow.keras.models import load_model
import code
import copy
import scipy.ndimage as sn
from PIL import Image
from tqdm import tqdm
from .model_simulated_RGB101 import get_testing_model_resnet101
from .human_seg.human_seg_gt import human_seg_combine_argmax


right_part_idx = [2, 3, 4,  8,  9, 10, 14, 16]
left_part_idx =  [5, 6, 7, 11, 12, 13, 15, 17]
human_part = [0,1,2,4,3,6,5,8,7,10,9,12,11,14,13]
human_ori_part = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
seg_num = 15 # current model supports 15 parts only

def recover_flipping_output(oriImg, heatmap_ori_size, paf_ori_size, part_ori_size):

    heatmap_ori_size = heatmap_ori_size[:, ::-1, :]
    heatmap_flip_size = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
    heatmap_flip_size[:,:,left_part_idx] = heatmap_ori_size[:,:,right_part_idx]
    heatmap_flip_size[:,:,right_part_idx] = heatmap_ori_size[:,:,left_part_idx]
    heatmap_flip_size[:,:,0:2] = heatmap_ori_size[:,:,0:2]

    paf_ori_size = paf_ori_size[:, ::-1, :]
    paf_flip_size = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))
    paf_flip_size[:,:,ori_paf_idx] = paf_ori_size[:,:,flip_paf_idx]
    paf_flip_size[:,:,x_paf_idx] = paf_flip_size[:,:,x_paf_idx]*-1

    part_ori_size = part_ori_size[:, ::-1, :]
    part_flip_size = np.zeros((oriImg.shape[0], oriImg.shape[1], 15))
    part_flip_size[:,:,human_ori_part] = part_ori_size[:,:,human_part]
    return heatmap_flip_size, paf_flip_size, part_flip_size

def recover_flipping_output2(oriImg, part_ori_size):

    part_ori_size = part_ori_size[:, ::-1, :]
    part_flip_size = np.zeros((oriImg.shape[0], oriImg.shape[1], 15))
    part_flip_size[:,:,human_ori_part] = part_ori_size[:,:,human_part]
    return part_flip_size

def part_thresholding(seg_argmax):
    background = 0.6
    head = 0.5
    torso = 0.8

    rightfoot = 0.55 
    leftfoot = 0.55
    leftthigh = 0.55
    rightthigh = 0.55
    leftshank = 0.55
    rightshank = 0.55
    rightupperarm = 0.55
    leftupperarm = 0.55
    rightforearm = 0.55
    leftforearm = 0.55
    lefthand = 0.55
    righthand = 0.55
    
    part_th = [background, head, torso, leftupperarm ,rightupperarm, leftforearm, rightforearm, lefthand, righthand, leftthigh, rightthigh, leftshank, rightshank, leftfoot, rightfoot]
    th_mask = np.zeros(seg_argmax.shape)
    for indx in range(15):
        part_prediction = (seg_argmax==indx)
        part_prediction = part_prediction*part_th[indx]
        th_mask += part_prediction

    return th_mask


def process (oriImg, flipImg, params, model_params, model):
    
    input_scale = 1.0
    multiplier = [x * model_params['boxsize'] / oriImg.shape[0] for x in params['scale_search']]
    seg_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 15))

    segmap_scale1 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))
    segmap_scale2 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))
    segmap_scale3 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))
    segmap_scale4 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))

    segmap_scale5 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))
    segmap_scale6 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))
    segmap_scale7 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))
    segmap_scale8 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))

    for m in range(len(multiplier)):
        scale = multiplier[m]*input_scale
        imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        pad = [ 0,
                0, 
                (imageToTest.shape[0] - model_params['stride']) % model_params['stride'],
                (imageToTest.shape[1] - model_params['stride']) % model_params['stride']
              ]
        
        imageToTest_padded = np.pad(imageToTest, ((0, pad[2]), (0, pad[3]), (0, 0)), mode='constant', constant_values=((0, 0), (0, 0), (0, 0)))

        input_img = imageToTest_padded[np.newaxis, ...]
        
        # print( "\tActual size fed into NN: ", input_img.shape)

        output_blobs = model.predict(input_img)
        seg = np.squeeze(output_blobs[2])
        seg = cv2.resize(seg, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                             interpolation=cv2.INTER_CUBIC)
        seg = seg[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        seg = cv2.resize(seg, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        if m==0:
            segmap_scale1 = seg
        elif m==1:
            segmap_scale2 = seg         
        elif m==2:
            segmap_scale3 = seg
        elif m==3:
            segmap_scale4 = seg


    # flipping
    for m in range(len(multiplier)):
        scale = multiplier[m]
        imageToTest = cv2.resize(flipImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        pad = [ 0,
                0, 
                (imageToTest.shape[0] - model_params['stride']) % model_params['stride'],
                (imageToTest.shape[1] - model_params['stride']) % model_params['stride']
              ]
        
        imageToTest_padded = np.pad(imageToTest, ((0, pad[2]), (0, pad[3]), (0, 0)), mode='constant', constant_values=((0, 0), (0, 0), (0, 0)))
        input_img = imageToTest_padded[np.newaxis, ...]
        # print( "\tActual size fed into NN: ", input_img.shape)
        output_blobs = model.predict(input_img)

        # extract outputs, resize, and remove padding
        seg = np.squeeze(output_blobs[2])
        seg = cv2.resize(seg, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                             interpolation=cv2.INTER_CUBIC)
        seg = seg[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        seg = cv2.resize(seg, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
        seg_recover = recover_flipping_output2(oriImg, seg)

        if m==0:
            segmap_scale5 = seg_recover
        elif m==1:
            segmap_scale6 = seg_recover         
        elif m==2:
            segmap_scale7 = seg_recover
        elif m==3:
            segmap_scale8 = seg_recover

    segmap_a = np.maximum(segmap_scale1,segmap_scale2)
    segmap_b = np.maximum(segmap_scale4,segmap_scale3)
    segmap_c = np.maximum(segmap_scale5,segmap_scale6)
    segmap_d = np.maximum(segmap_scale7,segmap_scale8)
    seg_ori = np.maximum(segmap_a, segmap_b)
    seg_flip = np.maximum(segmap_c, segmap_d)
    seg_avg = np.maximum(seg_ori, seg_flip)

    
    return seg_avg


def run_image(input_path:str, output_path:str):

    print('-' * 20)
    print('Downloading Weights File...\nPlease Wait...')
    print('-' * 20)
    weights_path = tf.keras.utils.get_file('cdcl.h5',
              					'https://github.com/Adeel-Intizar/Xtreme-Vision/releases/download/1.0/cdcl_model.h5',
                  			cache_subdir = 'weights/', cache_dir = 'xtreme_vision')


    print('start processing...')
    # load model
    model = get_testing_model_resnet101() 
    model.load_weights(weights_path)
    params, model_params = config_reader()

    scale_list = [1.0]

    params['scale_search'] = scale_list

    if input_path.endswith(".png") or input_path.endswith(".jpg"):
        
        Img = cv2.imread(input_path)
        flipImg = cv2.flip(Img, 1)
        oriImg = (Img / 256.0) - 0.5
        flipImg = (flipImg / 256.0) - 0.5
        seg = process(oriImg, flipImg, params, model_params, model)
        seg_argmax = np.argmax(seg, axis=-1)
        seg_max = np.max(seg, axis=-1)
        th_mask = part_thresholding(seg_argmax)
        seg_max_thres = (seg_max > 0.1).astype(np.uint8)
        seg_argmax *= seg_max_thres
        seg_canvas = human_seg_combine_argmax(seg_argmax)
        # cur_canvas = cv2.imread(image)
        canvas = cv2.addWeighted(seg_canvas, 0.6, Img, 0.4, 0)
        cv2.imwrite(output_path, canvas)


def run_video(input_path:str, output_path:str, fps:int = 25):

    print('-' * 20)
    print('Downloading Weights File...\nPlease Wait...')
    print('-' * 20)
    weights_path = tf.keras.utils.get_file('cdcl.h5',
              					'https://github.com/Adeel-Intizar/Xtreme-Vision/releases/download/1.0/cdcl_model.h5',
                  			cache_subdir = 'weights/', cache_dir = 'xtreme_vision')    

    
    # load model
    model = get_testing_model_resnet101() 
    model.load_weights(weights_path)
    params, model_params = config_reader()
    params['scale_search'] = [1.0]

    if not (input_path.endswith(".mp4") or input_path.endswith(".avi")):
        raise RuntimeError ('Invalid video format, Valid formats are mp4 and avi.')
    else:

        out = None
        cap = cv2.VideoCapture(input_path)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'\nThere are {length} Frames in this video')
        print('-' * 20)
        print('Detecting Objects in the Video... Please Wait...')
        print('-' * 20)

        while(cap.isOpened()):
            retreive, frame = cap.read()
            if not retreive:
                break

            flipImg = cv2.flip(frame, 1)
            oriImg = (frame / 256.0) - 0.5
            flipImg = (flipImg / 256.0) - 0.5

            seg = process(oriImg, flipImg, params, model_params, model)
            seg_argmax = np.argmax(seg, axis=-1)
            seg_max = np.max(seg, axis=-1)
            th_mask = part_thresholding(seg_argmax)
            seg_max_thres = (seg_max > 0.1).astype(np.uint8)
            seg_argmax *= seg_max_thres
            seg_canvas = human_seg_combine_argmax(seg_argmax)
            canvas = cv2.addWeighted(seg_canvas, 0.6, frame, 0.4, 0)

            if out is None:
                fourcc = cv2.VideoWriter_fourcc(*'DIVX')    
                out = cv2.VideoWriter(output_path, fourcc, fps, (frame.shape[1], frame.shape[0]))
            
            out.write(canvas)
        print('Done. Processing has been Finished... Please Check Output Video.')
        out.release()
        cap.release()