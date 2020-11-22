import numpy as np
import yaml
import cv2
import os
import sys
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import argparse
from utils.data_utils import Dataset
from utils.data_utils import DataPrepper
from models.darknet import DarkNet19
from models.yolo import YOLOv2
from models.losses import YOLO_SSE
from utils.dataset import *
from utils.evaluation import Evaluator

config = yaml.safe_load(open('config.yaml'))

class Tester:
    def __init__(self):
        self.ds = Dataset(config)
        self.ds.anchors,self.ds.anchors_coords = self.ds.get_anchors()
        pass
    
    def load_dataloader(self):
        root_loc = config['home_dirs'][1] if config['use_colab'] else config['home_dirs'][0]
        num_workers = 1 if config['use_gpu'] else config['num_workers']
        dataset = VOCDetection(root=root_loc + config['test_data_loc'])
        data_loader = torch.utils.data.DataLoader(
            dataset, 
            1,
            shuffle=config['shuffle_data'],
            num_workers=num_workers, 
            collate_fn=detection_collate,
            drop_last=True
        )
        return data_loader

    def open_model(self,model_loc,device):
        print("-- Opening Model --")
        root_loc = config['home_dirs'][1] if config['use_colab'] else config['home_dirs'][0]
        yolo = YOLOv2(config=config,mode='test',anchor_boxes=self.ds.anchors,device=device)
        yolo.load_state_dict(torch.load(root_loc+model_loc,map_location=torch.device('cpu')))
        yolo.eval()
        if(config['use_gpu'] and torch.cuda.is_available()):
            yolo = yolo.to(device)
        print("-- Successfully Loaded Model --")
        return yolo
    
    def training_optimizer(self,optimizer_name,model):
        if(optimizer_name == 'adam'):
            optimizer = torch.optim.Adam(
                params=model.parameters(),
                lr=config['d_learning_rate'],
                weight_decay=config['d_weight_decay']
            )
            return optimizer
        elif(optimizer_name == 'sgd'):
            optimizer = torch.optim.SGD(
                params=model.parameters(),
                lr=config['d_learning_rate'],
                momentum=config['d_momentum'],
                weight_decay=config['d_weight_decay']
            )
            return optimizer
    
    def load_checkpoint(self,checkpoint_loc):
        root_loc = config['home_dirs'][1] if config['use_colab'] else config['home_dirs'][0]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        yolo = YOLOv2(config=config,mode='test',anchor_boxes=self.ds.anchors,device=device)
        if(config['use_gpu'] and torch.cuda.is_available()):
            yolo = yolo.to(device)
        optimizer = self.training_optimizer(config['optimizer_type'],yolo)

        checkpoint = torch.load(root_loc+checkpoint_loc+'checkpoint.pt',map_location=device)
        yolo.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        total_loss = checkpoint['total_loss']
        conf_loss = checkpoint['conf_loss']
        localization_loss = checkpoint['bbox_loss']
        cls_loss = checkpoint['cls_loss']
        yolo.eval()
        return yolo,optimizer,epoch,total_loss,conf_loss,localization_loss,cls_loss

    def test_single_example(self,dataloader):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if(config['warmstart']):
            # Load latest checkpoint
            yolo,optimizer,curr_epoch,total_loss,conf_loss,localization_loss,cls_loss = self.load_checkpoint(config['checkpoints_loc'])
        else:
            yolo = self.open_model(config['model_loc'],device)
        criterion = YOLO_SSE(config=config,device=device)
        def testing_loop(dataloader):
            for idx,(images,targets) in enumerate(dataloader):
                targets = self.ds.generate_gt_data(targets)
                targets = torch.from_numpy(targets).float()
                if(config['use_gpu'] and torch.cuda.is_available()):
                    images = images.to(device)
                    targets = targets.to(device)
                # Get predictions on image from the model
                bbox_preds,cls_preds,conf_preds,preds = yolo(images,targets)
                images = images.squeeze(0)
                images = images.cpu().numpy()
                print(images.shape)
                self.examine_predictions(images,bbox_preds)
                break

            pass
        testing_loop(dataloader=dataloader)
        pass

    def test(self,dataloader):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if(config['warmstart']):
            # Load latest checkpoint
            yolo,optimizer,curr_epoch,total_loss,conf_loss,localization_loss,cls_loss = self.load_checkpoint(config['checkpoints_loc'])
        else:
            yolo = self.open_model(config['model_loc'],device)
        criterion = YOLO_SSE(config=config,device=device)
        def testing_loop(dataloader):
            predictions = []
            gt = []
            for idx,(images,targets) in enumerate(dataloader):
                gt.append(targets)
                targets = self.ds.generate_gt_data(targets)
                targets = torch.from_numpy(targets).float()
                if(config['use_gpu'] and torch.cuda.is_available()):
                    images = images.to(device)
                    targets = targets.to(device)
                # Get predictions on image from the model
                bbox_preds,cls_preds,conf_preds,preds = yolo(images,targets)
                predictions.append(preds)
                
                images = images.squeeze(0)
            return gt,predictions
        gt,predictions = testing_loop(dataloader=dataloader)
        
        # -- Evaluation --
        # print("--- Evaluation Results ---")
        # e = Evaluator(config,gt,predictions)
        # print(len(gt))
        # print(len(predictions))
        # e.mAP(gt,predictions)
        pass
    
    def unnormalize_img(self,img):
        img = img[:, :, ::-1].transpose(1,2,0)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img *= 255.0
        img = np.ascontiguousarray(img, dtype=np.uint8)
        return img

    def examine_predictions(self,img=None,bbox_preds=None,show_grid=True):
        img = self.unnormalize_img(img)
        
        x = np.floor(config['img_size'] / config['num_grid_cells'])
        y = np.floor(config['img_size'] / config['num_grid_cells'])
        move_x = x
        move_y = y

        fig,ax = plt.subplots(1)
        ax.imshow(img)

        # Shows the Feature map grid size
        if (show_grid):
            for b in range(1):
                for grid_row in range(config['img_size'] // config['grid_stride']):
                    plt.plot([0,config['img_size']],[move_y,move_y],color='y',marker='.')
                    for grid_col in range(config['img_size'] // config['grid_stride']):
                        plt.plot([move_x,move_x],[0,config['img_size']],color='y',marker='.')
                        move_x += x
                    move_x = x
                    move_y += y
        # Now add boxes
        for b in range(1):
            for idx,bbox in enumerate(bbox_preds):
                rect = patches.Rectangle(
                            (bbox[0]*x,bbox[1]*y),
                            (bbox[2] - bbox[0])*x ,(bbox[3] - bbox[1])*y,
                            linewidth=2,
                            edgecolor='r',
                            facecolor='none'
                        )
                ax.add_patch(rect)
        #plt.savefig(config['save_plots_loc']+'predictions_im2.png')

    def examine_predictions2(self,img,bbox_preds):
        img = self.unnormalize_img(img)
        
        x = np.floor(config['img_size'] / config['num_grid_cells'])
        y = np.floor(config['img_size'] / config['num_grid_cells'])
        move_x = x
        move_y = y

        fig,ax = plt.subplots(1)
        ax.imshow(img)
        for b in range(1):
            for grid_row in range(config['img_size'] // config['grid_stride']):
                plt.plot([0,config['img_size']],[move_y,move_y],color='y',marker='.')
                for grid_col in range(config['img_size'] // config['grid_stride']):
                    plt.plot([move_x,move_x],[0,config['img_size']],color='y',marker='.')

                    # Draw Anchors
                    for anchor in range(len(self.ds.anchors)):
                        # Draw Predicitions
                        # bbox = bxbybwbh
                        bbox = bbox_preds[b][grid_row + grid_col][anchor]

                        rect = patches.Rectangle(
                            (bbox[0]*x,bbox[1]*y),
                            bbox[2],bbox[3],
                            linewidth=2,
                            edgecolor='r',
                            facecolor='none'
                        )
                        ax.add_patch(rect)
                    move_x += x
                move_x = x
                move_y += y

        plt.savefig(config['save_plots_loc']+'predictions_im.png')

def detection_collate(batch):
    #print("Length of Batch = ", len(batch))
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        #print("---> ", sample[1])
        #targets.append(torch.FloatTensor(sample[1]))
        targets.append(sample[1])
    imgs = torch.stack(imgs, 0)
    return imgs,targets

def main():
    if(config['training_model'] == 'detector'):
        tester = Tester()
        data_loader = tester.load_dataloader()
        tester.test(data_loader)

if __name__ == "__main__":
    main()