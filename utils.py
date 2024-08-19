import torch
import math
from PIL import Image, ImageDraw, ImageFont
from torch.nn import functional as F
import logging
import os
import numpy as np
import json
import cv2
import random
colors_rgb = [
    (144, 238, 144),  # 浅绿色
    (255,165,0),
    (255, 127, 80),   # 珊瑚色
    (224, 255, 255),   # 浅青色
    (255, 218, 185),  # 淡橙色
    (255, 182, 193),  # 淡粉色
    (224, 255, 255),   # 浅青色 
    (230, 230, 250),
    (255, 192, 203),
    (211, 211, 211)

]

font = ImageFont.truetype('/home/hoc/disk/font/ArialBold.ttf', 30)

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
    
    draw = ImageDraw.Draw(horizontal_concatenated)
    if prompt is not None:
        draw.text((0, 0), prompt, font=font, fill=(0,0,0))
    if filename is not None:
        draw.text((45,45),filename,font=font, fill=(0,0,0))
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
                # 将数组中为零的元素替换为最小非零值
                dis_matrix+=nonzero_min
            dis_matrix = torch.tensor(cv2.resize(dis_matrix, (H, W), interpolation=cv2.INTER_LINEAR)).float().cuda()

                
            for obj_position in object_positions[obj_idx]:
                # print(obj_position)
                ca_map_obj = attn_map[:, :, obj_position].reshape(b, H, W)
                # print("ca",ca_map_obj)
                # print("ac:",(ca_map_obj / soft_attn).reshape(b, -1).sum(dim=-1))
                activation_value = (ca_map_obj / dis_matrix).reshape(b, -1).sum(dim=-1)/ca_map_obj.reshape(b, -1).sum(dim=-1)
                control_loss = torch.mean((1 - activation_value) ** 2)
                # print("control_loss:",control_loss)
                obj_loss += control_loss
        
                pre_activation_value = (ca_map_obj*dis_matrix).reshape(b, -1).sum(dim=-1)/ca_map_obj.reshape(b, -1).sum(dim=-1)
                move_loss = torch.mean((1-1/pre_activation_value)**2)*move_rate
                # print("move_loss:",move_loss)
                obj_loss += move_loss
            loss += (obj_loss/len(object_positions[obj_idx]))
            

    # print("up")
    for attn_map_integrated in attn_maps_up[0]:
        attn_map = attn_map_integrated
        #
        b, i, j = attn_map.shape
        H = W = int(math.sqrt(i))

        for obj_idx in range(object_number):
            obj_loss = 0
            dis_matrix = dis_matrixs[obj_idx]
            if (len(dis_matrix[dis_matrix <1e-9 ]) !=0):
                nonzero_min = dis_matrix[dis_matrix>1e-9].min()
                if (nonzero_min  <1e-9) :
                    nonzero_min = 0.001
                # 将数组中为零的元素替换为最小非零值
                dis_matrix+=nonzero_min
            
            dis_matrix = torch.tensor(cv2.resize(dis_matrix, (H, W), interpolation=cv2.INTER_LINEAR)).float().cuda()
            # cv2.imwrite('./down{}.png'.format(obj_idx),np.array(soft_attn.cpu()))
        
            for obj_position in object_positions[obj_idx]:
            
                ca_map_obj = attn_map[:, :, obj_position].reshape(b, H, W)
                activation_value = (ca_map_obj / dis_matrix).reshape(b, -1).sum(dim=-1)/ca_map_obj.reshape(b, -1).sum(dim=-1)
                control_loss = torch.mean((1 - activation_value) ** 2)
                # print("control_loss:",control_loss)
                obj_loss += control_loss
            
                pre_activation_value = (ca_map_obj*dis_matrix).reshape(b, -1).sum(dim=-1)/ca_map_obj.reshape(b, -1).sum(dim=-1)
                move_loss = torch.mean((1-1/pre_activation_value)**2)*move_rate
                # print("move_loss:",move_loss)
                obj_loss += move_loss
            loss += (obj_loss / len(object_positions[obj_idx]))
            
    loss = loss / (object_number * (len(attn_maps_up[0]) + len(attn_maps_mid)))
    return loss

def compute_ca_loss_points_diff(attn_maps_mid, attn_maps_up, points,object_positions):
    loss = 0
    object_number = len(points)
    if object_number == 0:
        return torch.tensor(0).float().cuda() if torch.cuda.is_available() else torch.tensor(0).float()
    
    for attn_map_integrated in attn_maps_mid:
        attn_map = attn_map_integrated
        b, i, j = attn_map.shape
        # print(attn_map.shape)
        H = W = int(math.sqrt(i))
        # print(H,W)
        for obj_idx in range(object_number):
            obj_loss = 0
            mask = torch.zeros(size=(H, W)).cuda() if torch.cuda.is_available() else torch.zeros(size=(H, W))
            for obj_point in points[obj_idx]:
                x_min, y_min, x_max, y_max = int((obj_point[0]-0.2) * W), \
                    int((obj_point[1]-0.2) * H), int((obj_point[0]+0.2) * W), int((obj_point[1]+0.2) * H)
                x_min = max(0,x_min)
                y_min = max(0,y_min)
                x_max = min(W,x_max)
                y_max = min(H,y_max)
                mask[y_min: y_max, x_min: x_max] = 1
                
            for obj_position in object_positions[obj_idx]:
                ca_map_obj = attn_map[:, :, obj_position].reshape(b, H, W)
                activation_value = (ca_map_obj * mask).reshape(b, -1).sum(dim=-1)/ca_map_obj.reshape(b, -1).sum(dim=-1)
                # print(activation_value)
                obj_loss += torch.mean((1 - activation_value) ** 2)
            loss += (obj_loss/len(object_positions[obj_idx]))
            

    # print("up")
    for attn_map_integrated in attn_maps_up[0]:
        attn_map = attn_map_integrated
        #
        b, i, j = attn_map.shape
        # print(attn_map.shape)
        H = W = int(math.sqrt(i))

        for obj_idx in range(object_number):
            obj_loss = 0
            mask = torch.zeros(size=(H, W)).cuda() if torch.cuda.is_available() else torch.zeros(size=(H, W))
            for obj_point in points[obj_idx]:
                x_min, y_min, x_max, y_max = int((obj_point[0]-0.2) * W), \
                    int((obj_point[1]-0.2) * H), int((obj_point[0]+0.2) * W), int((obj_point[1]+0.2) * H)
                x_min = max(0,x_min)
                y_min = max(0,y_min)
                x_max = min(W,x_max)
                y_max = min(H,y_max)
                mask[y_min: y_max, x_min: x_max] = 1


            for obj_position in object_positions[obj_idx]:
            
                ca_map_obj = attn_map[:, :, obj_position].reshape(b, H, W)
                # ca_map_obj = attn_map[:, :, object_positions[obj_position]].reshape(b, H, W)

                activation_value = (ca_map_obj * mask).reshape(b, -1).sum(dim=-1) / ca_map_obj.reshape(b, -1).sum(
                    dim=-1)

                obj_loss += torch.mean((1 - activation_value) ** 2)
            loss += (obj_loss / len(object_positions[obj_idx]))
            
    loss = loss / (object_number * (len(attn_maps_up[0]) + len(attn_maps_mid)))
    
    return loss

#check
def merge_attention_map(attn_maps_down,attn_maps_mid, attn_maps_up):
    attn_maps_list = []
    for attn_map_integrated_down,attn_map_integrated_up in zip(attn_maps_down,attn_maps_up[::-1]):
        attn_maps = []
        for att_map in attn_map_integrated_down:
            attn_maps.append(att_map)
        for att_map in attn_map_integrated_up:
            attn_maps.append(att_map)
        attn_maps = torch.stack(attn_maps).mean(0)
        attn_maps_list.append(attn_maps)
    for attn_map_integrated in attn_maps_mid:
        attn_map = attn_map_integrated
        attn_maps_list.append(attn_map)
    return attn_maps_list    

def get_attention_map(ca_attn_maps,self_attn_maps,object_positions,self_res = 32,ca_res = 16):
    object_number = len(object_positions)

    if object_number == 0:
        return torch.tensor(0).float().cuda() if torch.cuda.is_available() else torch.tensor(0).float()
    b,i, j = ca_attn_maps.shape
    # for ca_attn_map_integrated,self_attn_map_integrated in zip(ca_attn_maps,self_attn_maps):
    #     b,i, j = ca_attn_map_integrated.shape
       
    #     H = W = int(math.sqrt(i))
    #     if (H == self_res):
    #         self_attn_map = self_attn_map_integrated
    #     if (H == ca_res):
    #         ca_attn_map = ca_attn_map_integrated
    masks = []
    for obj_idx in range(object_number):
        index = object_positions[obj_idx]
        
        if isinstance(index, list):
            ca = ca_attn_maps[:,:, index].mean(-1) #有个开始符号 token 从1开始
        elif isinstance(index, int):
            ca = ca_attn_maps[:,:, index]

        # print(affinity_mat.shape)
        ca = ca.reshape(b,ca_res,ca_res).mean(0)
        ca = ca - ca.min()
        ca = ca/ca.max()

        # idrx = (ca>0.3)&(ca<0.6)
        # ca[idrx] +=0.2
        masks.append(ca.cpu().numpy())
      
    return masks
def get_attention_mask(ca_attn_maps,self_attn_maps,points,object_positions,self_res = 32,ca_res = 16):
    
    object_number = len(object_positions)
    if object_number == 0:
        return torch.tensor(0).float().cuda() if torch.cuda.is_available() else torch.tensor(0).float()

    outs = []

    for ca_attn_map_integrated,self_attn_map_integrated in zip(ca_attn_maps,self_attn_maps):
        b,i, j = ca_attn_map_integrated.shape
       
        H = W = int(math.sqrt(i))
        if (H == self_res):
            self_attn_map = self_attn_map_integrated
        if (H == ca_res):
            ca_attn_map = ca_attn_map_integrated
        
    affinity_mat = torch.matrix_power(self_attn_map.mean(0), 4)

    # 有问题 多个点对应一个组合token ！！！！！！！！！！！！！！！！！！！！！！！！！
    for obj_idx in range(object_number):
        index = object_positions[obj_idx]
        
        if isinstance(index, list):
            ca = ca_attn_map[:,:, index].mean(-1) #有个开始符号 token 从1开始
        
        elif isinstance(index, int):
            ca = ca_attn_map[:,:, index]
    
        # print(affinity_mat.shape)
        ca = ca.reshape(b,ca_res,ca_res)
        ca = F.interpolate(ca.unsqueeze(0), (self_res, self_res), mode="bicubic")[0].mean(0)
        
        out = ( affinity_mat@ca.reshape(self_res**2, 1)).reshape(self_res, self_res)
        
        out = out - out.min()
        out = out / out.max()
        outs.append(out)
    
    outs = torch.stack(outs)
    up_outs = F.interpolate(outs.unsqueeze(0), (512, 512), mode="bicubic")[0].cpu().numpy()

    
    outs_max = up_outs.max(axis=0)

    mask = np.zeros((512, 512), dtype=np.uint8)

    valid = outs_max >= 0.6
    mask[valid] = (up_outs.argmax(axis=0) + 1)[valid]
   
    softmax_mask = []

    for i in range(object_number):
        s_mask = np.zeros((512, 512), dtype=np.uint8)
        indics = mask == i+1
        s_mask[indics] = up_outs[i][indics]
        softmax_mask.append(s_mask)
        

    return mask,softmax_mask
                
   
def find_tensor_positions(tensor, sub_tensor):
    positions = []
    for i in range(tensor.shape[0] - sub_tensor.shape[0] + 1):
        if torch.equal(tensor[i:i+sub_tensor.shape[0]], sub_tensor):
            positions = [i for i in range(i,i+sub_tensor.shape[0])]
            return positions
    return positions

def Pharse2idx_tokenizer(prompt, phrases,tokenizer):
    phrases = [x.strip() for x in phrases.split(';')]
    print(phrases)
    phrase_ids = []
    object_positions = []
    print(prompt)
    text_input = tokenizer(prompt, padding="max_length",return_length=True, return_overflowing_tokens=False, max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")



    for phrase in phrases:
        phrase_ids.append(tokenizer(phrase, return_length=True, return_overflowing_tokens=False, max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids)
    for id in phrase_ids:
        position = find_tensor_positions(text_input.input_ids[0],id[0][1:-1])
        object_positions.append(position)
    

    print(object_positions)
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

def get_points_masks(points,res = 512):
    masks = []
    for i,point in enumerate(points):
        mask = np.ones((res, res), dtype=np.uint8)

        for j in range(len(point) - 1):
            point_1 = (int(point[j][0]*512),int(point[j][1]*512))
            point_2 = (int(point[j+1][0]*512),int(point[j+1][1]*512))
            cv2.line(mask, point_1, point_2, 0, 1)
        distance_transform = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        mask = np.array(distance_transform)
        masks.append(mask)
    return masks


def draw_mask(masks):
    pil = Image.new('RGB', (512, 512), color=(64, 64, 64))
    pil = np.array(pil)
    for i,mask in enumerate(masks):
        indics = np.where(mask==1)
        pil[indics] = colors_rgb[i]
    return Image.fromarray(pil)