import time

from pytriton.client import ModelClient

def sendto_A100_SAM(cv2_image,model="SAM_h"):
    
    # be sure to convert the image first before sending as inference!
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    url = "http://202.92.132.48:8000"
    
    with ModelClient(url, model) as client:
        
        
        t = time.time()
        outputs = client.infer_sample(cv2_image)
        print(f"[SAM] Elapsed: {time.time() - t}")
        
        segmentation = outputs['segmentation']
        masks = []
        
        for i in range(segmentation.shape[0]):
            masks.append({
                'segmentation': segmentation[i,:,:],
                'area': outputs['area'][i],
                "bbox": outputs['bbox'][i],
                "predicted_iou": outputs['predicted_iou'][i],
                "stability_score": outputs['stability_score'][i],
            })
        
        return masks
