import numpy as np
import sys
import torch 
import torch.nn as nn
import torch.nn.functional as F
from utils.load_config import open_config
#np.set_printoptions(threshold=sys.maxsize)
#torch.set_printoptions(profile="full")

class YOLO_SSE(nn.Module):
    """
    YOLO Sum-Squared Error Loss Function as
    proposed in: https://arxiv.org/abs/1506.02640
    with alterations to work with YOLOv2

    This loss function consists of three sub-functions that 
    calculate 3 losses based on the output predictions from YOLO.
    """
    def __init__(self,config,anchor_boxes,device):
        super(YOLO_SSE2,self).__init__()
        self.config = open_config()
        self.anchor_boxes = anchor_boxes
        self.device = device

        self.num_grid_cells = self.config['num_grid_cells']

        self.grid_xy = self.create_grid(self.num_grid_cells**2)
        self.grid_xy = self.grid_xy.to(self.device)

        if(anchor_boxes != None):
            self.anchor_boxes = torch.as_tensor(anchor_boxes).float()
        self.anchor_wh = self.anchor_boxes.repeat(self.num_grid_cells**2, 1, 1).unsqueeze(0).to(self.device)
        pass

    def create_grid(self,img_size):
        """
        Arguments:
            img_size: This is the final feature map size (H*W), which is input_size // stride
                      for example if input_size is 416, then final feat map is 13x13, so 169  
        Returns:
            grid_xy: 4D Tensor of indexed coordinates of grid [B,]
        """
        grid_w,grid_h = np.sqrt(img_size),np.sqrt(img_size)
        x_grid,y_grid = torch.arange(grid_w),torch.arange(grid_h)
        y_grid,x_grid = torch.meshgrid(y_grid,x_grid)

        # [img_size,img_size,2] ex: [13,13,2]
        grid_xy = torch.stack([x_grid, y_grid], dim=-1).float()
        # Adds needed dimensions for addition into bbox predictions
        grid_xy = grid_xy.unsqueeze(0).unsqueeze(3)
        # Reshapes grid into [B,img_size*img_size,1,2]
        grid_xy = grid_xy.view(grid_xy.shape[0],grid_xy.shape[1]*grid_xy.shape[2],grid_xy.shape[3],grid_xy.shape[4])
        return grid_xy
    
    def convert_bboxes(self,txtytwth,requires_grad):
        """
        Conversion of coordinate space of bounding boxes
        from txtytwth raw predictions to x1y1x2y2 format.

        Note: Predictions that are output are unnormalized.
        """
        B,S,N,coords = txtytwth.shape[0],np.sqrt(txtytwth.shape[1]),txtytwth.shape[2],txtytwth.shape[-1]
        assert(txtytwth.shape == (B,S*S,N,4))
        def txtytwth_2_bxbybwbh(txtytwth):
            bxby = torch.sigmoid(txtytwth[...,:2]) + self.grid_xy
            bwbh = self.anchor_wh * torch.exp(txtytwth[...,2:])
            bxbybwbh = torch.cat([bxby,bwbh],dim=-1)
            return bxbybwbh
        def bxbybwbh_2_x1y1x2y2(bxbybwbh,requires_grad):
            x1y1x2y2 = torch.zeros_like(bxbybwbh,requires_grad=requires_grad)
            bxbybwbh[...,0:2] *= 32
            x1y1x2y2[...,0] = (bxbybwbh[...,0] - bxbybwbh[...,2]/2)
            x1y1x2y2[...,1] = (bxbybwbh[...,1] - bxbybwbh[...,3]/2)
            x1y1x2y2[...,2] = (bxbybwbh[...,0] + bxbybwbh[...,2]/2)
            x1y1x2y2[...,3] = (bxbybwbh[...,1] + bxbybwbh[...,3]/2)
            return x1y1x2y2
        exists = torch.where(
            txtytwth[...,:] > 0,
            torch.ones_like(txtytwth[...,:]),
            torch.zeros_like(txtytwth[...,:])
        )
        bxbybwbh = txtytwth_2_bxbybwbh(txtytwth)
        x1y1x2y2 = bxbybwbh_2_x1y1x2y2(bxbybwbh,requires_grad)
        return x1y1x2y2
    
    def generate_masks(self,ground_truth):
        gt_conf = ground_truth[:,:,:,:1].to(self.device)
        #####################################################
        #                  Exists Object                    #
        #####################################################
        obj_mask = torch.zeros_like(gt_conf).to(self.device)
        # contains 1 for any box that had a obj
        obj_mask = torch.where(gt_conf == 1.0,torch.ones_like(gt_conf),obj_mask).float().to(self.device)
        #####################################################
        #                   No Object                       #
        #####################################################
        noobj_mask = torch.zeros_like(gt_conf).to(self.device)
        # contains 1 for any box that didnt have an obj
        noobj_mask = torch.where(gt_conf == 0.0,torch.ones_like(gt_conf),noobj_mask).float().to(self.device)
        return obj_mask,noobj_mask
    
    def localization_loss(self,ground_truth,bbox_predictions,obj_mask,pred_obj_mask):
        l_coord = self.config['l_coord'] #5
        eps = self.config['loss_eps']

        gt_bbox = ground_truth[:,:,:,1:5].to(self.device)
        gt_bbox = obj_mask * self.convert_bboxes(gt_bbox,False)
        bbox_predictions = pred_obj_mask * self.convert_bboxes(bbox_predictions,False)

        gt_bbox /= 416
        bbox_predictions /= 416

        #####################################################
        #                   XY Loss                         #
        #####################################################
        gt_x_bbox = gt_bbox[:,:,:,:1].to(self.device)
        gt_y_bbox = gt_bbox[:,:,:,1:2].to(self.device)

        pred_x_bbox = bbox_predictions[:,:,:,:1].to(self.device)
        pred_y_bbox = bbox_predictions[:,:,:,1:2].to(self.device)

        x_sq_diff = (torch.abs(gt_x_bbox - pred_x_bbox))**2
        y_sq_diff = (torch.abs(gt_y_bbox - pred_y_bbox))**2

        xy_loss = torch.sum( obj_mask * ( x_sq_diff + y_sq_diff),dtype=torch.float32)

        #####################################################
        #                   WH Loss                         #
        #####################################################
        gt_w_bbox = (gt_bbox[:,:,:,2:3] - gt_bbox[:,:,:,:1]).to(self.device)
        gt_h_bbox = (gt_bbox[:,:,:,3:4] - gt_bbox[:,:,:,:1:2]).to(self.device)

        pred_w_bbox = (bbox_predictions[:,:,:,2:3] - bbox_predictions[:,:,:,:1]).to(self.device)
        pred_h_bbox = (bbox_predictions[:,:,:,3:4] - bbox_predictions[:,:,:,1:2]).to(self.device)

        w_sq_diff = (torch.sqrt(torch.abs(gt_w_bbox)+eps) - torch.sqrt(torch.abs(pred_w_bbox)+eps))**2
        h_sq_diff = (torch.sqrt(torch.abs(gt_h_bbox)+eps) - torch.sqrt(torch.abs(pred_h_bbox)+eps))**2

        wh_loss = torch.sum(obj_mask * (w_sq_diff + h_sq_diff),dtype=torch.float32)
        
        bbox_loss = ((l_coord * xy_loss) + (l_coord * wh_loss))
        return bbox_loss

    def confidence_loss(self,ground_truth,conf_predictions):
        # Down weight the boxes that didn't contain the object
        l_noobj = self.config['l_noobj'] # 0.5
        gt_conf = ground_truth[:,:,:,:1].to(self.device)
        conf_predictions = torch.sigmoid(conf_predictions).to(self.device)

        assert(gt_conf.shape == conf_predictions.shape)

        #####################################################
        #                Object Loss                        #
        #####################################################
        obj_mask = torch.zeros_like(conf_predictions).to(self.device)
        # contains 1 for any box that had a obj
        obj_mask = torch.where(gt_conf == 1.0,torch.ones_like(conf_predictions),obj_mask).float().to(self.device)
        obj_loss = torch.sum(obj_mask * (gt_conf - conf_predictions)**2,dtype=torch.float32)
        #####################################################
        #             No Object Loss                        #
        #####################################################
        noobj_mask = torch.zeros_like(conf_predictions).to(self.device)
        # contains 1 for any box that didnt have an obj
        noobj_mask = torch.where(gt_conf == 0.0,torch.ones_like(conf_predictions),noobj_mask).float().to(self.device)
        noobj_loss = l_noobj * (torch.sum(noobj_mask * (gt_conf - conf_predictions)**2,dtype=torch.float32))
        
        conf_loss = obj_loss + noobj_loss
        return conf_loss

    def classification_loss(self,ground_truth,cls_predictions):
        use_personal_calc = True
        gt_conf = ground_truth[:,:,:,:1].to(self.device)
        gt_cls = ground_truth[:,:,:,5:6].to(self.device)

        gt_conf = gt_conf.view(gt_conf.shape[0],gt_conf.shape[1] * gt_conf.shape[2],gt_conf.shape[3]).contiguous().to(self.device)
        gt_cls = gt_cls.view(gt_cls.shape[0],gt_cls.shape[1] * gt_cls.shape[2],gt_cls.shape[3]).contiguous().to(self.device)
        
        # [B,S*S,N,Preds] -> [B,S*S*N,Preds] i.e. [1,169,5,4] -> [1,845,4]
        cls_predictions = cls_predictions.view(
            cls_predictions.shape[0],
            cls_predictions.shape[1] * cls_predictions.shape[2],
            cls_predictions.shape[3]).contiguous().to(self.device)
        
        cls_predictions = F.softmax(cls_predictions,dim=-1)
        # _, [B,S*S*N]
        max_prediction,max_pred_idx = torch.max(cls_predictions,dim=-1)
        cls_predictions_1 = torch.zeros((
            cls_predictions.shape[0],
            cls_predictions.shape[1],
            1)).to(self.device)
        batch_idx = torch.arange(0,cls_predictions.shape[0]).to(self.device)
        gc_idx = torch.arange(0,cls_predictions.shape[1]).to(self.device)
        #cls_predictions_1[batch_idx,gc_idx,0] = cls_predictions[batch_idx,gc_idx,max_pred_idx]
        
        max_pred_idx = torch.flatten(max_pred_idx) # B * S * S * N
        cls_predictions_1 = torch.flatten(cls_predictions_1)
        # Grabbing indicies of the maxes classes
        cls_predictions_1[:] = max_pred_idx[:]#torch.flatten(max_prediction)[:]
        cls_predictions_1 = cls_predictions_1.view(
            cls_predictions.shape[0],
            cls_predictions.shape[1],
            1).contiguous()
        if(use_personal_calc):
            cls_loss = torch.sum(gt_conf * (gt_cls - cls_predictions_1)**2) #torch.sum(gt_conf * torch.sum((gt_cls - cls_predictions_1)**2,1))
        else:
            cls_loss_function = nn.MSELoss(reduction='sum')
            cls_predictions_1 = cls_predictions_1.view(cls_predictions_1.shape[0]*cls_predictions_1.shape[1],-1).contiguous()
            gt_cls = gt_cls.view(gt_cls.shape[0]*gt_cls.shape[1],-1).contiguous()#torch.flatten(gt_cls.view(gt_cls.shape[0]*gt_cls.shape[1],-1).contiguous())
            gt_conf = gt_conf.view(gt_conf.shape[0]*gt_conf.shape[1],-1).contiguous()
            cls_loss = cls_loss_function(gt_conf * cls_predictions_1.float() ,gt_conf * gt_cls.long())
        return cls_loss

    def forward(self,ground_truth,bbox_predictions,conf_predictions,cls_predictions):
        total_loss,conf_loss,bbox_loss,cls_loss = 0,0,0,0
        # [B,13,13,5,6] -> [B,169,5,4]
        ground_truth = ground_truth.view(
            ground_truth.shape[0],
            ground_truth.shape[1]*ground_truth.shape[2],
            ground_truth.shape[3],
            ground_truth.shape[4]).contiguous().to(self.device)
        
        # Get Object and No Object Masks
        obj_mask,noobj_mask = self.generate_masks(ground_truth)
        pred_obj_mask = torch.where(conf_predictions >= 0.5,torch.ones_like(conf_predictions),torch.zeros_like(conf_predictions))

        bbox_loss = self.localization_loss(ground_truth,bbox_predictions,obj_mask,pred_obj_mask)
        conf_loss = self.confidence_loss(ground_truth,conf_predictions)
        cls_loss = self.classification_loss(ground_truth,cls_predictions)

        total_loss = conf_loss + bbox_loss + cls_loss
        return total_loss, conf_loss, bbox_loss, cls_loss