import numpy as np
import torch
import torch.utils.data
import yaml
import cv2
import json
import xml
from xml.etree import ElementTree
import os
import sys
import re
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

config = yaml.safe_load(open("config.yaml"))


class Dataset:

    def __init__(self,config):
        pass

    def atoi(self,text):
        return int(text) if text.isdigit() else text

    def natural_keys(self,text):
        return [ self.atoi(c) for c in re.split(r'(\d+)', text) ]

    def open_traffic_ds(self,config):
        imgs = []
        annots = []
        # Get images
        for idx,im_file in (enumerate(sorted(os.listdir(config["ds_loc"]+"images/"),key=self.natural_keys))):
            if(config['data_count'] == 'full'):
                xml_file = im_file.split(".")[0] + ".xml"
                tree = ElementTree.parse(config["ds_loc"]+"annotations/"+xml_file)
                im_data = get_xml_data(tree)

                img = cv2.imread(config["ds_loc"]+"images/"+im_file)
                imgs.append(img)
                annots.append(im_data)
            else:
                if(idx == config['data_count']):
                    break
                xml_file = im_file.split(".")[0] + ".xml"
                tree = ElementTree.parse(config["ds_loc"]+"annotations/"+xml_file)
                im_data = self.get_xml_data(tree)

                img = cv2.imread(config["ds_loc"]+"images/"+im_file)
                imgs.append(img)
                annots.append(im_data)
        
        return imgs,annots

    def get_xml_data(self,tree):
        root = tree.getroot()
        bboxes = []
        file_name = ''
        w = 0
        h = 0
        x_min = 0
        x_max = 0
        y_min = 0
        y_max = 0
        im_data = {
            'filename': file_name,
            'w' : 0,
            'h' : 0,
            'bboxes': bboxes
        }
        for item in root:
            if(item.tag == 'filename'):
                im_data['filename'] = item.text
            if(item.tag == "size"):
                im_data['w'] = int(item[0].text)
                im_data['h'] = int(item[1].text)
            if(item.tag == "object"):
                for obj_items in item:
                    if(obj_items.tag == "bndbox"):
                        bbox_data = {
                            'x_min' : 0,
                            'x_max' : 0,
                            'y_min' : 0,
                            'y_max' : 0,
                            'x_mid' : 0,
                            'y_mid' : 0
                        }
                        bbox_data['x_min'] = int(obj_items[0].text)
                        bbox_data['y_min'] = int(obj_items[1].text)
                        bbox_data['x_max'] = int(obj_items[2].text)
                        bbox_data['y_max'] = int(obj_items[3].text)
                        
                        # Calculating the mid-point of the bbox
                        bbox_data['x_mid'] = np.floor( 1/2 * (bbox_data['x_min'] + bbox_data['x_max']) )
                        bbox_data['y_mid'] = np.floor( 1/2 * (bbox_data['y_min'] + bbox_data['y_max']) )
                        bboxes.append(bbox_data)
        im_data['bboxes'] = bboxes
        return im_data
    
    def get_anchors(self):
        anchors = None
        if(config['use_colab']):
            root_loc = config['home_dirs'][1]
        else:
            root_loc = config['home_dirs'][0]
        with open(root_loc + 'anchors.json','r') as anchor_file:
            anchors = json.load(anchor_file)
        anchors_coords = []
        anchors_wh = []
        for anchor_name,anchor in anchors.items():
            anchors_coords.append([anchor['x_min'],anchor['y_min'],anchor['x_max'],anchor['y_max']])
            anchors_wh.append([anchor['x_max'],anchor['y_max']])
        return anchors_wh, anchors_coords
    
    def reformat_bbox(self,bbox):
        # Transform the bbox coords back to image space
        x_min = bbox[0] * config['img_size']
        y_min = bbox[1] * config['img_size']
        x_max = bbox[2] * config['img_size']
        y_max = bbox[3] * config['img_size']
        bbox_w = x_max - x_min
        bbox_h = y_max - y_min
        
        # Calculate the center point
        x_mid = 1/2 * (x_max + x_min)
        y_mid = 1/2 * (y_max + y_min)

        # Find the anchor box that best matches the ground truth box
        gt_bbox_origin = np.array([0,0,bbox_w,bbox_h])
        ious = []
        for anchor in self.anchors_coords:
            iou = DataPrepper.iou(self,bbox1=anchor,bbox2=gt_bbox_origin,input_type='list')
            ious.append(iou)        
        best_anchor_idx = np.argmax(ious)
        best_iou = ious[best_anchor_idx]

        # map it to the grid cell coordinate space 
        c_x_s = x_mid / config['grid_stride']
        c_y_s = y_mid / config['grid_stride']
        bbox_ws = bbox_w / config['grid_stride']
        bbox_hs = bbox_h / config['grid_stride']

        # get grid locations
        grid_x = int(c_x_s)
        grid_y = int(c_y_s)

        # Prepare coordinates
        p_w = self.anchors[best_anchor_idx][0]
        p_h = self.anchors[best_anchor_idx][1]

        tx = c_x_s - grid_x
        ty = c_y_s - grid_y
        tw = np.log( bbox_w / p_w )
        th = np.log( bbox_h / p_h )

        return grid_x, grid_y, tx, ty, tw, th, best_anchor_idx

    def generate_gt_data(self,target):
        """
        target: List containing bbox and class
        """
        #[B, im_w*im_h, num_anchors, data]
        gt_data = np.zeros(
            (
                config['batch_size'],
                config['num_grid_cells'],
                config['num_grid_cells'],
                config['num_anchors'],
                (4 + 1 + 1)
            )
        )
        for batch_item_idx,item in enumerate(target):
            for bbox_idx,bbox in enumerate(item):
                grid_x, grid_y, tx, ty, tw, th, anchor_idx = self.reformat_bbox(bbox)
                gt_data[batch_item_idx,grid_y,grid_x,anchor_idx,0] = 1
                gt_data[batch_item_idx,grid_y,grid_x,anchor_idx,1] = tx
                gt_data[batch_item_idx,grid_y,grid_x,anchor_idx,2] = ty
                gt_data[batch_item_idx,grid_y,grid_x,anchor_idx,3] = tw
                gt_data[batch_item_idx,grid_y,grid_x,anchor_idx,4] = th
                gt_data[batch_item_idx,grid_y,grid_x,anchor_idx,5] = bbox[-1]
        return gt_data

class DataPrepper:
    
    def __init__(self, x_data=None, y_data=None):
        self.x_data = x_data
        self.y_data = y_data
        self.anchor_boxes = self.get_anchors()
        pass
    
    def get_anchors(self):
        with open('anchors.json','r') as anchor_file:
            return json.load(anchor_file)
        pass
    
    def rescale_data(self,x_data,y_data):
        """
        In order for the model to learn the locations of the objects
        we need to rescale the images and bounding boxes to the same
        height and width
        
        Due to the fact that the data is given in varying dimensions,
        we need to set it to a set size
        """
        
        for idx,img in enumerate(x_data):
            y = y_data[idx]
            
            # Resize img
            im_resized = cv2.resize(img,(config['img_size'],config['img_size']))
            x_data[idx] = im_resized
            
            # Rescale y_data accordingly
            x_scaler = config['img_size'] / y['w']
            y_scaler = config['img_size'] / y['h']
            
            y['w'] = config['img_size']
            y['h'] = config['img_size']
            
            # Rescale each bbox
            for bbox in y['bboxes']:
                bbox['x_min'] = int(( bbox['x_min'] * x_scaler ))
                bbox['x_max'] = int(( bbox['x_max'] * x_scaler ))

                bbox['y_min'] = int(( bbox['y_min'] * y_scaler ))
                bbox['y_max'] = int(( bbox['y_max'] * y_scaler ))

                bbox['x_mid'] = int(( bbox['x_mid'] * x_scaler ))
                bbox['y_mid'] = int(( bbox['y_mid'] * y_scaler ))
            
            y_data[idx] = y
        return x_data, y_data
        
    
    def normalize(self,x_data,y_data):
        for idx,img in enumerate(x_data):
            y = y_data[idx]
            
            # normalize img
            img = img[:, :, ::-1].transpose(0,1,2)
            img = np.ascontiguousarray(img, dtype=np.float32)
            img /= 255.0
            
            y['w'] = config['img_size']/config['img_size']
            y['h'] = config['img_size']/config['img_size']
            
            # normalize bboxes
            for bbox in y['bboxes']:
                bbox['x_min'] /= config['img_size']
                bbox['x_max'] /= config['img_size']

                bbox['y_min'] /= config['img_size']
                bbox['y_max'] /= config['img_size']

                bbox['x_mid'] /= config['img_size']
                bbox['y_mid'] /= config['img_size']
            
            x_data[idx] = img
            y_data[idx] = y
        return x_data, y_data
    
    def draw_grid(self,img,y_data):
        fig,ax = plt.subplots(1)
        # Visualize the grid cells that YOLO will be using to track an objects location
        x = np.floor(y_data['w'] / config['num_grid_cells'])
        y = np.floor(y_data['h'] / config['num_grid_cells'])

        print(x)
        print(y)

        move_x = x
        move_y = y
        #if(config['jupyter']):
        ax.imshow(img)
        for i in range(config['num_grid_cells']):
            plt.plot([move_x,move_x],[0,y_data['h']],color='r',marker='.')
            plt.plot([0,y_data['w']],[move_y,move_y],color='r',marker='.')
            move_x += x
            move_y += y
        if(config['jupyter']):
            plt.show()
        else:
            plt.savefig(config['save_plots_loc']+'grid_im.png')
            #plt.imsave(config['save_plots_loc']+'grid_im.png',img)
        
    def visualize_data(self,img,data):
        print(data['h'])
        print(data['w'])
        fig,ax = plt.subplots(1)
        if(config['jupyter']):
            ax.imshow(img)
        
        for bbox in data['bboxes']:
            rect = patches.Rectangle(
                (bbox['x_min'],bbox['y_min']),
                bbox['x_max']-bbox['x_min'],
                bbox['y_max']-bbox['y_min'],
                linewidth=2,
                edgecolor='r',
                facecolor='none'
            )
            pt = plt.plot(bbox['x_mid'],bbox['y_mid'],color='r',marker='.')
            ax.add_patch(rect)
        if(config['jupyter']):
            plt.show()
        else:
            plt.imsave(config['save_plots_loc']+'visualize.png',img)
        plt.cla()
        plt.clf()
        plt.close()
    
    def format_dataset(self,x_data,y_data):
        formatted_y_data = []
        for idx,img in enumerate(x_data):
            y = y_data[idx]
            formatted_boxes = []
            for bbox in y['bboxes']:
                formatted_boxes.append(np.array([bbox['x_min'],bbox['y_min'],bbox['x_max'],bbox['y_max']]))
            formatted_y_data.append(np.array(formatted_boxes))
        formatted_y_data = np.array(formatted_y_data)
        return x_data,formatted_y_data
    
    def compare_anchors(self,anchor_boxes,bbox):
        # Translate the anchor boxes to the ground truth box
        translated_anchors = self.translate_anchors(anchor_boxes,bbox)
        
        # Find the anchor box with the highest IOU
        best_anchor = {}
        greatest_iou = 0
        for idx,(a_box_name,a_box) in enumerate(translated_anchors.items()):
            iou = self.iou(bbox,a_box)
            if(iou > greatest_iou):
                greatest_iou = iou
                best_anchor = a_box
        return (idx,best_anchor)
    
    def translate_anchors(self,anchor_boxes,bbox):
        """
        Since the anchor boxes are located at the origin
        when they were read, we need to translate them to the location
        of the object's location
        :param: anchor_boxes - dict of anchor boxes 
        :param: bbox - dictionary that contains the point locations
        """
        shifted_anchors = {}
        for anchor_name,anchor in anchor_boxes.items():
            shifted_anchors[anchor_name] = {}
            if(bbox['x_mid'] > anchor['x_mid']):
                x_translate = bbox['x_min'] - anchor['x_min']
                # Anchor box need to shift right
                shifted_anchors[anchor_name]['x_min'] = anchor['x_min'] + x_translate
                shifted_anchors[anchor_name]['x_max'] = anchor['x_max'] + x_translate
            elif(bbox['x_mid'] < anchor['x_mid']):
                # Anchor box needs to shift left
                x_translate = anchor['x_min'] - bbox['x_min']
                shifted_anchors[anchor_name]['x_min'] = anchor['x_min'] - x_translate
                shifted_anchors[anchor_name]['x_max'] = anchor['x_max'] - x_translate

            if(bbox['y_mid'] > anchor['y_mid']):
                # Anchor box needs to shift up
                y_translate = bbox['y_min'] - anchor['y_min']
                shifted_anchors[anchor_name]['y_min'] = anchor['y_min'] + y_translate
                shifted_anchors[anchor_name]['y_max'] = anchor['y_max'] + y_translate
            elif(bbox['y_mid'] < anchor['y_mid']):
                # Anchor box needs to shift down
                y_translate = bbox['y_min'] - anchor['y_min']
                shifted_anchors[anchor_name]['y_min'] = anchor['y_min'] - y_translate
                shifted_anchors[anchor_name]['y_max'] = anchor['y_max'] - y_translate
        return shifted_anchors
        
    def iou(self,bbox1,bbox2,input_type='dict',visualize=False,debug=False):

        if(input_type == 'dict'):
            max_min_x = max(bbox1['x_min'],bbox2['x_min'])
            min_max_x = min(bbox1['x_max'],bbox2['x_max'])
            intersection_x = min_max_x - max_min_x
            
            max_min_y = max(bbox1['y_min'],bbox2['y_min'])
            min_max_y = min(bbox1['y_max'],bbox2['y_max'])
            intersection_y = min_max_y - max_min_y
            
            intersection_area = intersection_x * intersection_y
            if(intersection_x <= 0 or intersection_y <= 0 or intersection_area <= 0):
                iou = 0
            else:
                # Calculate Area of each box
                bbox1_area = (bbox1['x_max'] - bbox1['x_min']) * (bbox1['y_max'] - bbox1['y_min'])
                bbox2_area = (bbox2['x_max'] - bbox2['x_min']) * (bbox2['y_max'] - bbox2['y_min'])

                iou = (intersection_area)/(bbox1_area + bbox2_area - intersection_area)
            return iou
        elif(input_type == 'list'):
            max_min_x = max(bbox1[0],bbox2[0])
            min_max_x = min(bbox1[2],bbox2[2])
            intersection_x = min_max_x - max_min_x
            
            max_min_y = max(bbox1[1],bbox2[1])
            min_max_y = min(bbox1[3],bbox2[3])
            intersection_y = min_max_y - max_min_y
            
            intersection_area = intersection_x * intersection_y
            if(intersection_x <= 0 or intersection_y <= 0 or intersection_area <= 0):
                iou = 0
            else:
                # Calculate Area of each box
                bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
                bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

                iou = (intersection_area)/(bbox1_area + bbox2_area - intersection_area)
            return iou
    
    def generate_grid_cell_data(self,bbox,best_anchor,num_anchors):
        anchor_box_details = []
        for ab_idx in range(num_anchors):
            if(ab_idx == best_anchor[0]):
                anchor_box_details.append([1,bbox['x_mid'],bbox['y_min'],bbox['x_max'],bbox['y_max']])
            else:
                anchor_box_details.append([0,-1,-1,-1,-1])
        anchor_box_details = np.array(anchor_box_details).flatten()
        return anchor_box_details
    
    def format_data(self,x_data,y_data):
        """
        For each img y = 7 x 7 x 2 x (5) -> 7 x 7 x 10
        y_gi = [ci,x_min,y_min,x_max,y_max]
        """
        y_data_f = []
        for idx,img in enumerate(x_data):
            y = y_data[idx]
            
            x_grid_len = np.floor( (y['w'] * config['img_size']) / config['num_grid_cells'])
            y_grid_len = np.floor( (y['h'] * config['img_size']) / config['num_grid_cells'])
            
            if(config['DEBUG_MODE']):
                print("Filename = ", y['filename'])
                print("x_grid_len = ", x_grid_len)
                print("y_grid_len = ", y_grid_len)
            
            back_x = 0
            top_y = 0
            front_x = x_grid_len
            bottom_y = y_grid_len
            
            # This will hold our grid
            y_grid = []
            # Going thru each grid cell in img
            for grid_count in range(config['num_grid_cells'] * config['num_grid_cells']):
                
                if(config['DEBUG_MODE']):
                    print("Grid count = ", grid_count)
                        
                # Check each bbox to see if it's mid-point is within the grid cell
                for idx,bbox in enumerate(y['bboxes']):
                    if(config['DEBUG_MODE']):
                        print("bbox idx = ", idx)
                    if(idx + 1 == len(y['bboxes']) and 
                       ((bbox['x_mid'] * config['img_size'] > front_x or back_x > bbox['x_mid'] * config['img_size']) or 
                       (bbox['y_mid'] * config['img_size'] > bottom_y or top_y > bbox['y_mid'] * config['img_size']))):                        
                        # We know in these cases, the mid point cannot be in this grid box 
                        # because we know the object isn't present here, the bbox coords are set to -1
                        empty_grid_cell_array = []
                        for ab_idx in range(len(self.anchor_boxes)):
                            empty_grid_cell_array.append([0,-1,-1,-1,-1])
                        empty_grid_cell_array = np.array(empty_grid_cell_array).flatten()
                        y_gi = empty_grid_cell_array#np.array([0,bbox['x_mid'],bbox['y_min'],bbox['x_max'],bbox['y_max']])
                        y_grid.append(y_gi)
                        break
                        
                    elif(back_x <= bbox['x_mid'] * config['img_size'] and bbox['x_mid'] * config['img_size'] <= front_x and 
                       top_y <= bbox['y_mid']* config['img_size'] and bbox['y_mid'] * config['img_size'] <= bottom_y):
                        # This will be where the object is present
                        # Compare each anchor box with object
                        best_anchor = self.compare_anchors(self.anchor_boxes,bbox)
                        # Generate the y_gi array
                        y_gi = self.generate_grid_cell_data(bbox,best_anchor,len(self.anchor_boxes))
                        y_grid.append(y_gi)
                        break

                # Grid Location Handling
                if(front_x + x_grid_len >= y['w'] * config['img_size'] and 
                   bottom_y + y_grid_len >= y['h'] * config['img_size']):
                    # We reached the last grid cell
                    # Nothing happens here, the loop ends
                    pass
                else:
                    if(front_x + x_grid_len > y['w'] * config['img_size']):
                        # We need to reset to the front of the img and move down
                        back_x = 0 
                        front_x = x_grid_len

                        top_y += y_grid_len
                        bottom_y += y_grid_len
                    else:
                        back_x += x_grid_len
                        front_x += x_grid_len
            
            y_grid = np.array(y_grid)
            if(config['DEBUG_MODE']):
                print(y_grid)
                print(y_grid.shape)
            y_grid = np.reshape(y_grid,(7,7,len(self.anchor_boxes) * 5))
            y_data_f.append(y_grid)
        
        print("-- Data Formatted! --")
        return y_data_f