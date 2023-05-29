import torch
import numpy as np
import cv2
import time

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# initialize Segment Anything

sam = None

def init_SAM(device_type = 'cuda'):
    
    global sam, sam_device
    
    print('Initializing SAM...')
    sam = sam_model_registry['vit_h'](checkpoint="sam_vit_h_4b8939.pth")
    sam.to(device=device_type)

def segmentation_inference(cv2image):
    
    global sam
    t = time.time()
    print("Generating local SAM inference...")
    predictor = SamAutomaticMaskGenerator(sam)
    masks = predictor.generate(cv2image)
    print('[SAM] Elapsed:',time.time() - t)
    return masks