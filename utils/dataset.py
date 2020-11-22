import os.path as osp
import os
import re
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import xml.etree.ElementTree as ET

import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

config = yaml.safe_load(open("config.yaml"))

VOC_CLASSES = ( 'trafficlight','speedlimit','crosswalk','stop')
VOC_ROOT = "datasets/archive/"

class VOCAnnotationTransform:
    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()

            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                #cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]

class VOCDetection(data.Dataset):
    def __init__(self, root,transform=None, target_transform=VOCAnnotationTransform()):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self._annopath = osp.join(self.root, 'annotations','road%s.xml')
        self._imgpath = osp.join(self.root, 'images', 'road%s.png')
        self.ids = list()
        
        for idx,name in (enumerate(sorted(os.listdir(self.root +"images/"),key=self.natural_keys))):
            data_id = int(name.split('road')[1].split('.')[0])
            self.ids.append(data_id)
    
    def atoi(self,text):
        return int(text) if text.isdigit() else text

    def natural_keys(self,text):
        return [ self.atoi(c) for c in re.split(r'(\d+)', text) ]
    
    def rescale(self,img,y_data,width,height):
        # Resize img
        img = cv2.resize(img,(config['img_size'],config['img_size']))
        # Rescale y_data accordingly
        x_scaler = config['img_size'] / width
        y_scaler = config['img_size'] / height
        
        # Rescale each bbox
        for idx,bbox in enumerate(y_data):
            y_data[idx][0] = int(( bbox[0] * x_scaler )) # x_min
            y_data[idx][2] = int(( bbox[2] * x_scaler )) # x_max

            y_data[idx][1] = int(( bbox[1] * y_scaler )) # y_min
            y_data[idx][3] = int(( bbox[3] * y_scaler )) # y_max
        return img, y_data, width, height
    
    def normalize(self,img,y_data,width,height):
        """
        img is passed in as BGR Format
        """
        mean = config['ds_mean']
        std = config['ds_std']

        # BGR -> RGB
        img = img[:, :, ::-1]#.transpose(0,1,2)

        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        # # Subtract data by the mean
        # img[...,0] -= mean[0]
        # img[...,1] -= mean[1]
        # img[...,2] -= mean[2]

        # # Divide by std
        # img[...,0] /= std[0]
        # img[...,1] /= std[1]
        # img[...,2] /= std[2]
        
        width = config['img_size']/config['img_size']
        height = config['img_size']/config['img_size']

        # normalize bboxes
        for idx,bbox in enumerate(y_data):
            y_data[idx][0] = y_data[idx][0] / config['img_size']
            y_data[idx][1] = y_data[idx][1] / config['img_size']
            y_data[idx][2] = y_data[idx][2] / config['img_size']
            y_data[idx][3] = y_data[idx][3] / config['img_size']
                    
        return img, y_data, width, height

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)
        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]

        annot_data = ET.parse(self._annopath % img_id)
        #print(annot_data.find('filename').text)
        target = annot_data.getroot()
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)
        if self.transform is not None:
            target = np.array(target,dtype='float')
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        else:
            target = np.array(target,dtype=np.float32)
            labels = target[:, 4]
            img,boxes,width, height = self.rescale(img,target[:, :4],width,height)
            img,boxes,width,height = self.normalize(img,target[:, :4],width,height)
            if(index == 22):
                print("BOXES NOW = ",boxes)
            # # to rgb
            # img = img[:, :, (2, 1, 0)]
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1).float(), target, height, width
        # return torch.from_numpy(img), target, height, width