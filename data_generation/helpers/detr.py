
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection
import torchvision
import os

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder=None, processor=None, train=True):
        if img_folder:
            ann_file = os.path.join(img_folder, "detr_format.json")
            super(CocoDetection, self).__init__(img_folder, ann_file)
        self.processor = processor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        # feel free to add data augmentation here before passing them to the next step
        img, target = super(CocoDetection, self).__getitem__(idx)

        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.processor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        target = encoding["labels"][0] # remove batch dimension

        return pixel_values, target
  
    def process_img(self, pil_img):
        encoding = self.processor(images=pil_img, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension

        return pixel_values.cpu()

class DETRModel():
  def __init__(self, model_name):
    self.model = DetrForObjectDetection.from_pretrained(f"rdoshi21/{model_name}", id2label={0:"robotic gripper"})
    self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    self.processor = DetrImageProcessor.from_pretrained(f"rdoshi21/{model_name}")
    self.val_config = CocoDetection(img_folder='/global/scratch/users/riadoshi/vla/helpers/', processor=self.processor, train=False)

    

  def eval(self, images_np, batch_size, threshold=0.1):
    self.model.to(self.device)
    images = [Image.fromarray(img) for img in images_np]
    width, height = images[0].size

    arr = [self.val_config.process_img(image) for image in images]
    pixel_values = torch.cat([dd.unsqueeze(0) for dd in arr]).to(self.device)

    # batched inference
    postprocessed_outputs = []
    for i in range(0, len(images), batch_size):
      batch_pixel_vals = pixel_values[i:i+batch_size]
      with torch.no_grad():
        # forward pass to get class logits and bounding boxes
        output = self.model(pixel_values=batch_pixel_vals, pixel_mask=None)

      # output = {k: v.cpu() for k, v in output.items()}
      postprocessed_output = self.processor.post_process_object_detection(output,
                                                                target_sizes=[(height, width)]*batch_pixel_vals.shape[0],
                                                                threshold=threshold)
      postprocessed_outputs.extend(postprocessed_output)

    # get centroids from boxes
    centroids = []
    for (image, result) in zip(images, postprocessed_outputs):
        scores, labels, boxes = result['scores'], result['labels'], result['boxes']
        boxes = boxes.cpu()
        if len(scores)>0:
          best_match_idx = torch.argmax(scores).item()
          best_box = boxes[best_match_idx]
          (xmin, ymin, xmax, ymax) = best_box
          centroid = ((xmin.item() + xmax.item()) / 2, (ymin.item() + ymax.item()) / 2)
        else:
          centroid = None
        
        centroids.append(centroid)

    self.model.to('cpu') # move it back off the gpu
    torch.cuda.empty_cache()
    return centroids


