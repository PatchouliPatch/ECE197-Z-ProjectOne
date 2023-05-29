
import open_clip
import numpy as np
import torch
from PIL import Image
import cv2
import time

model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14',pretrained="datacomp_xl_s13b_b90k") # for GPU
tokenizer = open_clip.get_tokenizer('ViT-L-14')
mscoco_labels = None
device = None
text = None

def init_OpenCLIP(model_device = 'cpu'):
    
    global device, mscoco_labels, text
    # load OpenCLIP into memory:
    print("Intializing OpenCLIP...")
    device = torch.device(model_device)
    model.to('cuda') # load OpenCLIP
    
    with open("mscoco2017plus1_labels.txt",'r') as f:
        idx2label = eval(f.read())
    mscoco_labels = list(idx2label.values())
    text = tokenizer(mscoco_labels)
    text = text.to(device)
    
def cv2_to_PIL(cv2_image):
    #print(cv2_image)
    image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    return image

def local_openclip_inference(cv2_img): # modified to take openCV images
    
    global device, mscoco_labels, text
    
    # --------------------------------
    # Calculations and return values here for OpenCLIP
    # --------------------------------
    
    # tokenize the labels
    img = cv2_to_PIL(cv2_img) # convert to PIL image
    image = preprocess(img).unsqueeze(0)
    image = image.to(device)
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1) # calculates probabilities
        
    index = np.argmax(text_probs.cpu().numpy())
    
    #print({'label':mscoco_labels[index],'confidence':text_probs[0][index].numpy()}) # used for debugging
    
    return {'label':mscoco_labels[index],'confidence':text_probs[0][index].cpu().numpy(),'index':index}
    