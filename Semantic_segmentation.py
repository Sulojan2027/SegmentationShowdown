import os
import numpy as np
import cv2 as cv
import glob as glob

import tensorflow as tf
import tensorflow_hub as hub

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from zipfile import ZipFile
from urllib.request import urlretrieve

import warnings
import logging
import absl

# Interactive widgets
from ipywidgets import widgets
from io import BytesIO

dataset_path = '/kaggle/input/test-images'

fig, ax = plt.subplots(1, 4, figsize=(12,12))
images = []

for idx in range(4):
    # Reading image
    image = cv.imread(dataset_path + f"/camvid_sample_{idx+1}.png")
    
    # Convert image in BGR format to RGB.
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # Add a batch dimension which is required by the model.
    image_final = np.expand_dims(image_rgb, axis=0) / 255.0
    images.append(image_final)
    
    ax[idx].imshow(image)
    ax[idx].axis('off')
    ax[idx].set_title(f'Sample image {idx+1}')

plt.tight_layout()
plt.show()

class_index = \
    {
         0: [(64, 128, 64),  'Animal'],
         1: [(192, 0, 128),  'Archway'],
         2: [(0, 128, 192),  'Bicyclist'],
         3: [(0, 128, 64),   'Bridge'],
         4: [(128, 0, 0),    'Building'],
         5: [(64, 0, 128),   'Car'],
         6: [(64, 0, 192),   'Cart/Luggage/Pram'],
         7: [(192, 128, 64), 'Child'],
         8: [(192, 192, 128),'Column Pole'],
         9: [(64, 64, 128),  'Fence'],
        10: [(128, 0, 192),  'LaneMkgs Driv'],
        11: [(192, 0, 64),   'LaneMkgs NonDriv'],
        12: [(128, 128, 64), 'Misc Text'],
        13: [(192, 0, 192),  'Motorcycle/Scooter'],
        14: [(128, 64, 64),  'Other Moving'],
        15: [(64, 192, 128), 'Parking Block'],
        16: [(64, 64, 0),    'Pedestrian'],
        17: [(128, 64, 128), 'Road'],
        18: [(128, 128, 192),'Road Shoulder'],
        19: [(0, 0, 192),    'Sidewalk'],
        20: [(192, 128, 128),'Sign Symbol'],
        21: [(128, 128, 128),'Sky'],
        22: [(64, 128, 192), 'SUV/Pickup/Truck'],
        23: [(0, 0, 64),     'Traffic Cone'],
        24: [(0, 64, 64),    'Traffic Light'],
        25: [(192, 64, 128), 'Train'],
        26: [(128, 128, 0),  'Tree'],
        27: [(192, 128, 192),'Truck/Bus'],
        28: [(64, 0, 64),    'Tunnel'],
        29: [(192, 192, 0),  'Vegetation Misc'],
        30: [(0, 0, 0),      'Void'],
        31: [(64, 192, 0),   'Wall']  
    }

index = class_index == 0
print(index)

model_url = "https://tfhub.dev/google/HRNet/camvid-hrnetv2-w48/1"

seg_model = hub.load(model_url)
print("Model loaded!")

def class_to_rgb(mask_class, class_index):

    # RGB channels
    image_r = np.zeros_like(mask_class).astype('uint8')
    image_g = np.zeros_like(mask_class).astype('uint8')
    image_b = np.zeros_like(mask_class).astype('uint8')
    
    for class_id in range(len(class_index)):
        index = mask_class == class_id
        image_r[index] = class_index[class_id][0][0]
        image_g[index] = class_index[class_id][0][1]
        image_b[index] = class_index[class_id][0][2]
        
    seg_map_rgb = np.stack([image_r, image_g, image_b], axis=2)
    
    return seg_map_rgb

def image_overlay(image, seg_map_rgb):

    alpha = 1.0
    beta = 0.6
    gamma = 0.0
    
    image = (image*255.0).astype('uint8')
    seg_map_bgr = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    
    image = cv.addWeighted(image, alpha, seg_map_rgb, beta, gamma)
    image = cv.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image

def run_inference(images, model, class_index):
    fig, ax = plt.subplots(len(images), 3, figsize=(15,15))
    
    for idx,img in enumerate(images):
        
        pred_mask = model.predict(img).numpy()
        
        pred_mask = pred_mask[:,:,:,1:]
        pred_mask = np.squeeze(pred_mask)
        
        pred_mask_class = np.argmax(pred_mask, axis=-1)
        pred_mask_rgb = class_to_rgb(pred_mask_class, class_index)
        
        # Display the original image.
        ax[idx][0].imshow(img[0])
        ax[idx][0].set_title("Input Image")
        ax[idx][0].axis("off")

        # Display the predicted color segmentation mask.
        ax[idx][1].imshow(pred_mask_rgb)
        ax[idx][1].set_title("Predicted Mask")
        ax[idx][1].axis("off")

        # Display the predicted color segmentation mask overlayed on the original image.
        overlayed_image = image_overlay(img[0], pred_mask_rgb)
        ax[idx][2].imshow(overlayed_image)
        ax[idx][2].set_title("Overlayed Image")
        ax[idx][2].axis("off")
    plt.show()
    
    return 0

def plot_color_legend(class_index):
    # Extract colors and labels from class_index dictionary.
    color_array = np.array([[v[0][0], v[0][1], v[0][2]] for v in class_index.values()]).astype(np.uint8)
    class_labels = [val[1] for val in class_index.values()]

    fig, ax = plt.subplots(nrows=2, ncols=16, figsize=(20, 3))
    plt.subplots_adjust(wspace=0.5, hspace=0.01)

    # Display color legend.
    for i, axis in enumerate(ax.flat):
        axis.imshow(color_array[i][None, None, :])
        axis.set_title(class_labels[i], fontsize=8)
        axis.axis("off")


plot_color_legend(class_index)

run_inference(images, seg_model, class_index)
