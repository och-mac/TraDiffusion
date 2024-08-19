import torch
import math
from PIL import Image, ImageDraw, ImageFont
from torch.nn import functional as F
import logging
import os
import numpy as np
import cv2


colors = [
            (144, 238, 144),  
            (255,165,0),
            (255, 127, 80),   
            (255,0,0),
            (0,0,255)
        ]

def concat_images(images,prompt = None,filename = None):
    w = h = 0
    for image in images:
        w+=image.width
        h = image.height
    padding = 0
    if prompt is not None:
        padding = 80
    horizontal_concatenated = Image.new('RGB', (w, h+padding),(255,255,255))
    width = 0
    for image in images:
        horizontal_concatenated.paste(image, (width, padding))
        width+=image.width
    return horizontal_concatenated


def compute_ca_loss_masks(attn_maps_mid, attn_maps_up,dis_matrixs,object_positions,move_rate = 10):
    loss = 0
    object_number = len(object_positions)
    if object_number == 0:
        return torch.tensor(0).float().cuda() if torch.cuda.is_available() else torch.tensor(0).float()
    # print("len",object_number)
    for attn_map_integrated in attn_maps_mid:
        attn_map = attn_map_integrated
        b, i, j = attn_map.shape
        H = W = int(math.sqrt(i))
        for obj_idx in range(object_number):
            obj_loss = 0
            dis_matrix = dis_matrixs[obj_idx]
            if (len(dis_matrix[dis_matrix <1e-9 ]) !=0):
                nonzero_min = dis_matrix[dis_matrix>1e-9].min()
                if (nonzero_min  <1e-9) :
                    nonzero_min = 0.001
                dis_matrix+=nonzero_min
            dis_matrix = torch.tensor(cv2.resize(dis_matrix, (H, W), interpolation=cv2.INTER_LINEAR)).float().cuda()

                
            for obj_position in object_positions[obj_idx]:
                ca_map_obj = attn_map[:, :, obj_position].reshape(b, H, W)
                activation_value = (ca_map_obj / dis_matrix).reshape(b, -1).sum(dim=-1)/ca_map_obj.reshape(b, -1).sum(dim=-1)
                control_loss = torch.mean((1 - activation_value) ** 2)
                obj_loss += control_loss
        
                pre_activation_value = (ca_map_obj*dis_matrix).reshape(b, -1).sum(dim=-1)/ca_map_obj.reshape(b, -1).sum(dim=-1)
                move_loss = torch.mean((1-1/pre_activation_value)**2)*move_rate
                obj_loss += move_loss
            loss += (obj_loss/len(object_positions[obj_idx]))
            
    for attn_map_integrated in attn_maps_up[0]:
        attn_map = attn_map_integrated
        b, i, j = attn_map.shape
        H = W = int(math.sqrt(i))

        for obj_idx in range(object_number):
            obj_loss = 0
            dis_matrix = dis_matrixs[obj_idx]
            if (len(dis_matrix[dis_matrix <1e-9 ]) !=0):
                nonzero_min = dis_matrix[dis_matrix>1e-9].min()
                if (nonzero_min  <1e-9) :
                    nonzero_min = 0.001
                dis_matrix+=nonzero_min
            
            dis_matrix = torch.tensor(cv2.resize(dis_matrix, (H, W), interpolation=cv2.INTER_LINEAR)).float().cuda()        
            for obj_position in object_positions[obj_idx]:
            
                ca_map_obj = attn_map[:, :, obj_position].reshape(b, H, W)
                activation_value = (ca_map_obj / dis_matrix).reshape(b, -1).sum(dim=-1)/ca_map_obj.reshape(b, -1).sum(dim=-1)
                control_loss = torch.mean((1 - activation_value) ** 2)
                obj_loss += control_loss
            
                pre_activation_value = (ca_map_obj*dis_matrix).reshape(b, -1).sum(dim=-1)/ca_map_obj.reshape(b, -1).sum(dim=-1)
                move_loss = torch.mean((1-1/pre_activation_value)**2)*move_rate
                obj_loss += move_loss
            loss += (obj_loss / len(object_positions[obj_idx]))
            
    loss = loss / (object_number * (len(attn_maps_up[0]) + len(attn_maps_mid)))
    return loss


   
def find_tensor_positions(tensor, sub_tensor):
    positions = []
    for i in range(tensor.shape[0] - sub_tensor.shape[0] + 1):
        if torch.equal(tensor[i:i+sub_tensor.shape[0]], sub_tensor):
            positions = [i for i in range(i,i+sub_tensor.shape[0])]
            return positions
    return positions

def Pharse2idx_tokenizer(prompt, phrases,tokenizer):
    phrases = [x.strip() for x in phrases.split(';')]
    phrase_ids = []
    object_positions = []
    text_input = tokenizer(prompt, padding="max_length",return_length=True, return_overflowing_tokens=False, max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

    for phrase in phrases:
        phrase_ids.append(tokenizer(phrase, return_length=True, return_overflowing_tokens=False, max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids)
    for id in phrase_ids:
        position = find_tensor_positions(text_input.input_ids[0],id[0][1:-1])
        object_positions.append(position)
    return text_input,object_positions


def setup_logger(save_path, logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Create a file handler to write logs to a file
    file_handler = logging.FileHandler(os.path.join(save_path, f"{logger_name}.log"))
    file_handler.setLevel(logging.INFO)

    # Create a formatter to format log messages
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Set the formatter for the file handler
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    return logger


def points_to_masks(points,res=512):
    masks = []
    for i,point in enumerate(points):
        mask = np.ones((res, res), dtype=np.uint8)

        for j in range(len(point) - 1):
            point_1 = (int(point[j][0]*512),int(point[j][1]*512))
            point_2 = (int(point[j+1][0]*512),int(point[j+1][1]*512))
            cv2.line(mask, point_1, point_2, 0, 1)
        masks.append(mask)
    return masks

def masks_to_distances_matrixs(masks):
    distance_matrixs = []
    for map in masks:
        mask = np.where(map==1,0,1)
        distance_transform = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        distance_matrixs.append(distance_transform)
    return distance_matrixs


 