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

np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(profile="full")

config = yaml.safe_load(open("config.yaml"))

class Trainer:

    def __init__(self):
        self.ds = Dataset(config)
        self.ds.anchors,self.ds.anchors_coords = self.ds.get_anchors()
        pass

    
    def load_dataloader(self):
        root_loc = config['home_dirs'][1] if config['use_colab'] else config['home_dirs'][0]
        num_workers = 1 if config['use_gpu'] else config['num_workers']
        dataset = VOCDetection(root=root_loc + config['train_data_loc'])
        data_loader = torch.utils.data.DataLoader(
            dataset, 
            config['batch_size'],
            shuffle=config['shuffle_data'],
            num_workers=num_workers, 
            collate_fn=detection_collate,
            drop_last=True
        )
        return data_loader

    def classification_train(self,train_set):

        darknet19 = DarkNet19(config)
        optimizer = torch.optim.SGD(
            params=darknet19.parameters(),
            lr=config['d_learning_rate'],
            momentum=config['d_momentum'],
            weight_decay=config['d_weight_decay']
        )

        def train_loop(num_epochs,train_set):
            total_loss, conf_loss, localization_loss, cls_loss = 0,0,0,0
            print('=============== TRAINING STARTED =====================')
            for epoch in range(config['d_num_epochs']):
                for i,(images, targets) in enumerate(train_set):
                    optimizer.zero_grad()

                    optimizer.step()
                    pass
        
        train_loop(config['c_num_epochs'],train_set)
        pass

    def testing_method(self,train_set):
        
        print(len(train_set))

        def train_loop(num_epochs,train_set):
            total_loss, conf_loss, localization_loss, cls_loss = 0,0,0,0
            print('=============== TRAINING STARTED =====================')
            for epoch in range(config['d_num_epochs']):
                for i,(images, targets) in enumerate(train_set):
                    pass
        
        #train_loop(config['d_num_epochs'],train_set)

    def show_information(self,device):
        input_size = [config['img_size'],config['img_size']]
        training_info = {
            "Input size : " : input_size,
            "Batch size : " : config['batch_size'],
            "Learning Rate : " : config['d_learning_rate'],
            "Epochs : " : config['d_num_epochs'],
            "Device : " : device,
            "Pre-trained : " : config['use_pretrained'] 
        }
        for k,v in training_info.items():
            print(k,v)
        pass
    
    def training_optimizer(self,optimizer_name,model):
        if(optimizer_name == 'adam'):
            optimizer = torch.optim.Adam(
                params=model.parameters(),
                lr=config['d_learning_rate'],
                eps=config['optimizer_eps'],
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
        elif(optimizer_name == 'adagrad'):
            optimizer = torch.optim.Adagrad(
                params=model.parameters(), 
                lr=config['d_learning_rate'], 
                lr_decay=0, 
                weight_decay=0, 
                initial_accumulator_value=0, 
                eps=1e-10
            )
            return optimizer

    def detection_train(self,train_set):
        # -- Initializing primary vars -- 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        yolo = None
        optimizer = None
        curr_epoch = 0
        total_loss, conf_loss, localization_loss, cls_loss = 0,0,0,0

        # -- Getting Checkpointed Model or training from scratch? --
        if(config['warmstart']):
            # Load latest checkpoint
            yolo,optimizer,curr_epoch,total_loss,conf_loss,localization_loss,cls_loss = self.load_checkpoint(config['checkpoints_loc'])
        else:
            # Create YOLOv2 model with fresh weights
            yolo = YOLOv2(config,mode='train',anchor_boxes=self.ds.anchors,device=device)
            if(config['use_gpu'] and torch.cuda.is_available()):
                yolo = yolo.to(device)
            optimizer = self.training_optimizer(config['optimizer_type'],yolo)
            if(config['use_pretrained']):
                root_loc = config['home_dirs'][1] if config['use_colab'] else config['home_dirs'][0]
                # Loading backbone Darknet19 pretrained weights
                yolo.load_from_npz(root_loc + config['c_pretrained_weights_loc'])
        
        # -- YOLO Loss Fn for YOLOv2
        criterion = YOLO_SSE(config=config,anchor_boxes=self.ds.anchors,device=device)
        
        # -- Display Training Parameters -- 
        self.show_information(device)

        def train_loop(curr_epoch,num_epochs,train_set):
            print('=============== TRAINING STARTED =====================')
            for epoch in range(curr_epoch,curr_epoch+config['d_num_epochs']):
                epoch_total_loss,epoch_conf_loss,epoch_bbox_loss,epoch_cls_loss = 0,0,0,0
                
                # Forward pass thru each example in model
                for i,(images, targets) in tqdm(enumerate(train_set),total=len(train_set)):
                    # Zero out all gradients
                    optimizer.zero_grad()

                    # Transform ground truth data 
                    targets = self.ds.generate_gt_data(targets)
                    targets = torch.from_numpy(targets).float()

                    if(config['use_gpu'] and torch.cuda.is_available()):
                        images = images.to(device)
                        targets = targets.to(device)

                    # Get predictions on image from the model
                    bxbybwbh_preds,cls_preds,conf_preds = yolo(images,targets)

                    # Compare predictions and calculate the loss from the loss function
                    total_loss, conf_loss, localization_loss, cls_loss = criterion(targets,bxbybwbh_preds,conf_preds,cls_preds)
                    print("[Epoch {0}/{1}]: Total Loss: {2:.2f} | Conf Loss: {3:.2f} | BBox Loss: {4:.2f} | Cls Loss: {5:.2f}".format(
                    epoch + 1,
                    config['d_num_epochs'],
                    total_loss,conf_loss,localization_loss,cls_loss))

                    epoch_total_loss += total_loss
                    epoch_conf_loss += conf_loss
                    epoch_bbox_loss += localization_loss
                    epoch_cls_loss += cls_loss

                    # back prop
                    total_loss.backward()
                    if(config['clip_grad']):
                        torch.nn.utils.clip_grad_norm_(yolo.parameters(),max_norm=config['clip_grad_max_norm'])
                    optimizer.step()
                    #self.examine_predictions(images[0].data.numpy(),targets.data.numpy())
                print("==========================================================================================================")
                print("[Epoch {0}/{1}]: Total Loss: {2:.2f} | Conf Loss: {3:.2f} | BBox Loss: {4:.2f} | Cls Loss: {5:.2f}".format(
                    epoch + 1,
                    config['d_num_epochs'],
                    epoch_total_loss,epoch_conf_loss,epoch_bbox_loss,epoch_cls_loss))
                # -- Save Model --
                #self.save_model(yolo,epoch)

                # self.checkpoint_model(checkpoint_info={
                #     "model_state_dict" : yolo.state_dict(),
                #     "optimizer_state_dict" : optimizer.state_dict(),
                #     "epoch" : epoch,
                #     "total_loss" : total_loss,
                #     "conf_loss" : conf_loss,
                #     "bbox_loss" : localization_loss,
                #     "cls_loss" : cls_loss
                # })

                #break
                pass
        train_loop(curr_epoch,config['d_num_epochs'],train_set)
        pass

    def load_checkpoint(self,checkpoint_loc):
        root_loc = config['home_dirs'][1] if config['use_colab'] else config['home_dirs'][0]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        yolo = YOLOv2(config=config,mode='train',anchor_boxes=self.ds.anchors,device=device)
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
        yolo.train()
        return yolo,optimizer,epoch,total_loss,conf_loss,localization_loss,cls_loss

    def checkpoint_model(self,checkpoint_info):
        chck_pt_loc = config['checkpoints_loc']
        root_loc = config['home_dirs'][1] if config['use_colab'] else config['home_dirs'][0]
        save_loc = root_loc + chck_pt_loc
        model_name = "checkpoint.pt"
        torch.save(checkpoint_info,save_loc+model_name)
    
    def save_model(self,model,num_epoch=0):
        chck_pt_loc = config['checkpoints_loc']
        root_loc = config['home_dirs'][1] if config['use_colab'] else config['home_dirs'][0]
        save_loc = root_loc + chck_pt_loc
        model_name = "epoch_{}.pt".format(num_epoch)
        torch.save(model.state_dict(),save_loc+model_name)
        pass
    
    def unnormalize_img(self,img):
        mean = config['ds_mean']
        std = config['ds_std']

        img = img[:, :, :].transpose(1,2,0)

        img *= 255.0
        
        # # Multiply by std
        # img[...,0] *= std[0]
        # img[...,1] *= std[1]
        # img[...,2] *= std[2]

        # # Add data by the mean
        # img[...,0] += mean[0]
        # img[...,1] += mean[1]
        # img[...,2] += mean[2]

        img = np.ascontiguousarray(img, dtype=np.uint8)
        return img

    def examine_predictions(self,img,bxbybwbh_preds):
        img = self.unnormalize_img(img)
        
        x = np.floor(config['img_size'] / config['num_grid_cells'])
        y = np.floor(config['img_size'] / config['num_grid_cells'])
        move_x = x
        move_y = y

        fig,ax = plt.subplots(1)
        ax.imshow(img)
        for b in range(config['batch_size']):
            for grid_row in range(config['img_size'] // config['grid_stride']):
                plt.plot([0,config['img_size']],[move_y,move_y],color='y',marker='.')
                for grid_col in range(config['img_size'] // config['grid_stride']):
                    plt.plot([move_x,move_x],[0,config['img_size']],color='y',marker='.')
                    move_x += x
                move_x = x
                move_y += y
        
        for b in range(config['batch_size']):
            for grid_cell in range(169):
                    # Draw Anchors
                    for anchor in range(5):
                        # Draw Predicitions
                        bbox = bxbybwbh_preds[b,grid_cell,anchor]
                        
                        rect = patches.Rectangle(
                            (bbox[0],bbox[1]),
                            bbox[2]-bbox[0],bbox[3]-bbox[1],
                            linewidth=2,
                            edgecolor='r',
                            facecolor='none'
                        )
                        ax.add_patch(rect)
        plt.savefig(config['save_plots_loc']+'training_predictions_im.png')
        sys.exit()
        pass

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                0 dim
    """
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

def classification_collate(batch):
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        print("-->",sample[1][:,-1])
        targets.append(torch.FloatTensor(sample[1]))
    imgs = torch.stack(imgs, 0)
    return imgs,targets

def main():
    if(config['training_model'] == 'detector'):
        print("Detection")
        trainer = Trainer()
        data_loader = trainer.load_dataloader()
        trainer.detection_train(data_loader)
    elif(config['training_model'] == 'classifier'):
        """
        Use this when you're training DarkNet19 from scratch, otherwise you can load
        the pre-trained weights. Classification will only need to train solely on 
        the DarkNet19 network and will not alter any weights in YOLO. The paper
        suggests to train on 224 x 224 images
        """
        print("Classification")
        trainer = Trainer()
        data_loader = trainer.load_dataloader()
        trainer.classification_train(data_loader)
    elif(config['training_model'] == 'testing'):
        print("!! TESTING !!")
        dataset = VOCDetection(root="datasets/archive/")
        data_loader = torch.utils.data.DataLoader(
            dataset, 
            config['batch_size'],
            shuffle=config['shuffle_data'],
            collate_fn=classification_collate
        )
        trainer = Trainer()
        trainer.testing_method(data_loader)

    pass

if __name__ == '__main__':
    main()