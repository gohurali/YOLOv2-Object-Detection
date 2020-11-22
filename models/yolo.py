import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from models.darknet import DarkNet19
#np.set_printoptions(threshold=sys.maxsize)
#torch.set_printoptions(profile="full")
class YOLOv2(nn.Module):
    def __init__(self,config=None,mode='train',anchor_boxes=None,device='cpu'):
        super(YOLOv2, self).__init__()
        self.config = config
        self.mode = mode
        self.device = device
        
        if(anchor_boxes != None):
            self.anchor_boxes = torch.as_tensor(anchor_boxes).float()

        self.num_grid_cells = self.config['num_grid_cells']
        # Creating grid idx tensors
        self.num_grid_cells = self.config['img_size'] // self.config['grid_stride'] # 416 / 32
        self.grid_xy = self.create_grid(self.num_grid_cells**2)
        self.grid_xy = self.grid_xy.to(self.device)
        self.anchor_wh = self.anchor_boxes.repeat(self.num_grid_cells**2, 1, 1).unsqueeze(0).to(self.device)

        #####################################################
        #          Model Architecture Definition            #
        #####################################################
        
        self.backbone = DarkNet19(config)
        if(self.config['use_gpu'] and torch.cuda.is_available()):
            self.backbone.to(self.device)
        
        # Detection Layers
        self.dh_conv1 = nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=(3,3),stride=1,padding=1)
        self.dh_conv1_BN = nn.BatchNorm2d(1024)
        self.dh_conv2 = nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=(3,3),stride=1,padding=1)
        self.dh_conv2_BN = nn.BatchNorm2d(1024)

        # conv layer #18 from backbone
        self.re_route_layer = nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=(3,3),stride=1,padding=1)

        self.dh_conv3 = nn.Conv2d(in_channels=1024 * 5,out_channels=1024,kernel_size=(3,3),stride=1,padding=1)
        # Prediction layer
        self.dh_conv4 = nn.Conv2d(
            in_channels=1024,
            out_channels=config['num_anchors'] * (1 + 4 + config['num_classes']),
            kernel_size=(1,1),
            stride=1,
            padding=0
        )
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

    def reorg(self,input_tensor,stride,forward=False):
        """
        Reorg Layer as proposed by Joseph Redmon in his YoloV2 architecture
        this layer strides across the tensor by a given stride and grabs the
        given inputs

        Arguments:
            input_tensor: 4D tensor of shape [B,C,H,W]
            stride: Given stride of reorg layer
            forward: Boolean, False for forward prop(channel increase), True for backprop(channel decrease)
        Returns:
            4D Tensor of shape: [B,C,H,W]
        """  
        batch,channels,height,width = input_tensor.shape
        # Reshape 4D tensor into a flattened shape
        input_tensor = torch.reshape(input_tensor,(batch*channels*height*width,))
        
        # Prepare parameters for reorg of layer
        # calculate final sizes of feature map
        output_channels = channels * stride**2 #* stride * stride
        output_height = height // stride
        output_width = width // stride

        use_output_shape = True if output_channels >= channels else False

        C = output_channels if use_output_shape else channels
        H = output_height if use_output_shape else height
        W = output_width if use_output_shape else width

        sz1 = batch * channels * height * width
        sz2 = batch * C * H * W

        arrayLen = batch * C * H * W #len(input_tensor)
        arrayOut = np.zeros(arrayLen)
        out_c = C // stride**2 #(stride*stride)
        for b in range(batch):
            for k in range(C):
                for j in range(H):
                    for i in range(W):
                        in_index = i + W*(j + H*(k + C*b))
                        c2 = k % out_c
                        offset = k // out_c
                        w2 = i*stride + offset % stride
                        h2 = j*stride + offset // stride
                        out_index = int(w2 + W*stride*(h2 + H*stride*(c2 + out_c*b)))
                        if forward:
                            arrayOut[out_index] = input_tensor[in_index]
                        else:
                            arrayOut[in_index] = input_tensor[out_index]
        output_tensor = torch.from_numpy(arrayOut)
        output_tensor = torch.reshape(output_tensor,(batch,C,H,W))
        return output_tensor
    
    def reorg2(self,x,stride):
        """
        Reference:
        https://github.com/yjh0410/yolov2-yolov3_PyTorch/blob/master/utils/modules.py
        """
        batch_size, channels, height, width = x.size()
        height_s = height // stride
        width_s = width // stride
        
        x = x.view(batch_size, channels, height_s, stride, width_s, stride).transpose(3, 4).contiguous()
        x = x.view(batch_size, channels, height_s * width_s, stride * stride).transpose(2, 3).contiguous()
        x = x.view(batch_size, channels, stride * stride, height_s, width_s).transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, height_s, width_s)
        return x
    
    def decode_predictions(self,predictions):
        """
        Deconstruct the raw predictions from YOLO
        input_tensor: 4D Tensor of shape [B,C,H,W]

        Feature Maps calculation
        N = num anchors
        C = num classes
        N * (5 + C) = 5 * (5 + 4) = 5 * 9 = 45
        """
        batch_size,cells,preds = predictions.shape
        predictions = predictions.view(batch_size,cells,-1,4+1+self.config['num_classes']).contiguous()
        # [B, S*S, N, Pred] Pred, 1,4,num classes
        conf_preds = predictions[:,:,:,:1].to(self.device)
        txtytwth_preds = predictions[:,:,:,1:5].to(self.device)
        cls_preds = predictions[:,:,:,5:].to(self.device)
        return txtytwth_preds, cls_preds, conf_preds

    def txtytwth_2_bxbybwbh(self,txtytwth_preds):
        """
        Transforms the predictions to grid cell space
        txtytwth_preds: 4D tensor of shape [B,H*W,N,4] where N is # of AB's
                        H and W are final feature map size
        grid_xy shape -> [B,169,1,2] 
        Output: [B,S*S,N,4]         
        """
        batch_size,img_size,num_anchors,num_coords = txtytwth_preds.shape
        # bxby -> [B,169,5,2] you get the xy coords 
        bxby = torch.sigmoid(txtytwth_preds[:,:,:,:2]) + self.grid_xy   
        bwbh = self.anchor_wh * torch.exp(txtytwth_preds[:,:,:,2:])
        bxbybwbh = torch.cat([bxby,bwbh],dim=-1) 
        return bxbybwbh
    
    def bxbybwbh_2_x1y1x2y2(self,bxbybwbh,requires_grad):
        """
        Transforms the grid cell space coordinates to image space
        and also transforming to x_min,y_min,x_max,y_max

        Output: [B,S*S,N,4]
        """
        x1y1x2y2 = torch.zeros_like(bxbybwbh,requires_grad=requires_grad)
        bxbybwbh[...,0:2] *= 32
        x1y1x2y2[...,0] = (bxbybwbh[...,0] - bxbybwbh[...,2]/2)
        x1y1x2y2[...,1] = (bxbybwbh[...,1] - bxbybwbh[...,3]/2)
        x1y1x2y2[...,2] = (bxbybwbh[...,0] + bxbybwbh[...,2]/2)
        x1y1x2y2[...,3] = (bxbybwbh[...,1] + bxbybwbh[...,3]/2)
        return x1y1x2y2

    def score_iou(self,gt_boxes,predicted_boxes):
        """
        Calculate the IOU between the ground truth boxes and the 
        predicted boxes
        Input tensors should be of shape [B,S*S*N,4]
        where B is batch, S is final feature map size, and N is the
        number of anchor boxes used
        """
        gt_B,gt_img,gt_ab,gt_coords = gt_boxes.shape
        gt_boxes = gt_boxes.view(gt_B,gt_img*gt_ab,gt_coords).contiguous()
        pred_B,pred_img,pred_ab,pred_coords = predicted_boxes.shape
        predicted_boxes = predicted_boxes.view(pred_B,pred_img*pred_ab,pred_coords).contiguous()

        gt_boxes = gt_boxes.to(self.device)
        predicted_boxes = predicted_boxes.to(self.device)

        max_min_x = torch.max(gt_boxes[...,0],predicted_boxes[...,0])
        min_max_x = torch.min(gt_boxes[...,2],predicted_boxes[...,2])
        intersection_x = min_max_x - max_min_x

        max_min_y = torch.max(gt_boxes[...,1],predicted_boxes[...,1])
        min_max_y = torch.min(gt_boxes[...,3],predicted_boxes[...,3])
        intersection_y = min_max_y - max_min_y

        intersection_area = intersection_x * intersection_y

        gt_bbox_area = (gt_boxes[...,2] - gt_boxes[...,0]) * (gt_boxes[...,3] - gt_boxes[...,1])
        pred_bbox_area = (predicted_boxes[...,2] - predicted_boxes[...,0]) * (predicted_boxes[...,3] - predicted_boxes[...,1])

        iou = (intersection_area)/(gt_bbox_area + pred_bbox_area - intersection_area)
        return iou
    
    def get_best_box(self,txtytwth_preds,iou,S):
        iou = iou.view(self.config['batch_size'],S*S,self.config['num_anchors'],-1).contiguous()
        max_iou,max_iou_idx = torch.max(iou, dim=-2)
        best_pred_txtytwth = torch.zeros((
            self.config['batch_size'],
            S*S,
            5,
            4)).to(self.device)
        best_pred_txtytwth.to(self.device)
        batch_idxs = torch.arange(0,self.config['batch_size']).to(self.device)
        gc_idxs = torch.arange(0,S*S).to(self.device)
        best_ab_idx = torch.flatten(max_iou_idx[:,:,0]).to(self.device) # B * S * S * 1
        best_ab_idx = best_ab_idx.to(self.device)
        #best_pred_txtytwth[batch_idxs,gc_idxs,best_ab_idx,:] = txtytwth_preds[batch_idxs,gc_idxs,best_ab_idx,:]
        
        best_pred_txtytwth[:,:,best_ab_idx,:] = txtytwth_preds[:,:,best_ab_idx,:]

        #for idx in best_ab_idx:
        #    best_pred_txtytwth[:,:,idx,:] = txtytwth_preds[:,:,idx,:]
        return best_pred_txtytwth
        
    def filter_by_best_box(self,txtytwth_preds,cls_preds,conf_preds,iou,S):
        """
        Filter all predictions by the best boxes.
        This function first filters out all but the highest IOU box for each grid cell.
        Then, it filters all boxes but boxes with a confidence IOU of 0.5 or higher.

        It would be more efficient to go straight to the second part and filter 
        all boxes but >= 0.5 IOU, however, for debugging purposes, I went with this.
        """
        # [B,S*S,N,1]
        iou = iou.view(self.config['batch_size'],S*S,self.config['num_anchors'],-1).contiguous()
        
        # Get the Max IOU for the current grid cell and the index
        max_iou,max_iou_idx = torch.max(iou, dim=-2)
        best_box_predictions = torch.zeros((
            self.config['batch_size'],
            S*S,
            5,
            1+4+self.config['num_classes'])).to(self.device)
        best_box_predictions.to(self.device)

        best_ab_idx = torch.flatten(max_iou_idx[:,:,0]).to(self.device) # B * S * S * 1 
        best_ab_idx = best_ab_idx.to(self.device)

        grid_cell_idx = torch.arange(S*S)
        for idx,i in enumerate(best_ab_idx):
            best_box_predictions[:,grid_cell_idx,best_ab_idx[idx],0:1] = conf_preds[:,grid_cell_idx,best_ab_idx[idx],0:1]
            best_box_predictions[:,grid_cell_idx,best_ab_idx[idx],1:5] = txtytwth_preds[:,grid_cell_idx,best_ab_idx[idx],:]
            best_box_predictions[:,grid_cell_idx,best_ab_idx[idx],5:] = cls_preds[:,grid_cell_idx,best_ab_idx[idx],:]

        # best_box_predictions[:,grid_cell_idx,best_ab_idx,0:1] = conf_preds[:,grid_cell_idx,best_ab_idx,0:1]
        # best_box_predictions[:,grid_cell_idx,best_ab_idx,1:5] = txtytwth_preds[:,grid_cell_idx,best_ab_idx,:]
        # best_box_predictions[:,grid_cell_idx,best_ab_idx,5:] = cls_preds[:,grid_cell_idx,best_ab_idx,:]

        # Filter by IOU Threshold
        filtered = torch.zeros((
            self.config['batch_size'],
            S*S,
            5,
            1+4+self.config['num_classes'])).to(self.device)

        filtered = torch.where(
            best_box_predictions[...,0:1] >= 0.5,
            best_box_predictions,
            filtered
        )

        return filtered

    def forward_pass(self,x):
        """
        General forward pass of the YOLOv2 Layers
        """
        x_B,x_C,x_H,x_W = x.shape
        dn_cls_preds, conv_block_5, conv_block_6 = self.backbone(x)

        # Run block 6 thru detection layers
        conv_block_6 = F.leaky_relu(self.dh_conv1_BN(self.dh_conv1(conv_block_6)))
        conv_block_6 = F.leaky_relu(self.dh_conv2_BN(self.dh_conv2(conv_block_6)))

        # Route block 5 thru re-route layer
        conv_block_5 = self.re_route_layer(conv_block_5)
        # Now we need to reorg conv_block_5 result
        batch_size,C,H,W = conv_block_5.shape
        conv_block_5 = self.reorg2(conv_block_5,self.config['reorg_stride'])
        #conv_block_5 = self.reorg(conv_block_5,self.config['reorg_stride'],False)

        # Combine block 5 with block 6 to provide finer details
        detection_block = torch.cat([conv_block_5,conv_block_6],dim=1).to(self.device)
        detection_block = F.leaky_relu(self.dh_conv3(detection_block))
        preds = self.dh_conv4(detection_block)
        
        B, abC, H, W = preds.shape
        # [B, 45,13,13] -> [B, 169, 45]
        preds = preds.permute(0, 2, 3, 1).contiguous().view(B, H*W, abC).to(self.device)
        # Now we need to decode the feature maps and obtain the boxes
        #txtytwth_preds,cls_preds,conf_preds = self.decode_predictions(preds)
        txtytwth_preds,cls_preds,conf_preds = self.decode_predictions(preds)
        # -- Sending outputs to current device -- 
        txtytwth_preds = txtytwth_preds.to(self.device)
        return txtytwth_preds,cls_preds,conf_preds
    
    def training_forward(self,x,gt_targets):
        """
        Performs further operations needed for training and obtaining the best box
        """
        scale = torch.from_numpy(np.array([[x.shape[-1], x.shape[-1], x.shape[-1], x.shape[-1]]]))
        txtytwth_preds,cls_preds,conf_preds = self.forward_pass(x)

        # General Shape
        B,S,N = txtytwth_preds.shape[0],int(np.sqrt(txtytwth_preds.shape[1])),txtytwth_preds.shape[2]

        # Convert Predictions to (cxs,cys,H,W) -> (x_min,y_min,x_max,y_max)
        bxbybwbh_preds = self.txtytwth_2_bxbybwbh(txtytwth_preds)
        x1y1x2y2_preds = self.bxbybwbh_2_x1y1x2y2(bxbybwbh_preds,True)

        x1y1x2y2_preds = x1y1x2y2_preds.to(device=self.device)

        # [B,S,S,N,N*(1+4+ #Classes)] -> [B,S*S,N,N*(1+4+ #Classes)]
        gt_B,gt_S,gt_S,gt_N,gt_data = gt_targets.shape
        gt_targets = gt_targets.view(gt_B,gt_S*gt_S,gt_N,gt_data).contiguous()

        exists = torch.where(
            gt_targets[...,1:5] > 0,
            torch.ones_like(gt_targets[...,1:5]),
            torch.zeros_like(gt_targets[...,1:5])).to(device=self.device)

        # Convert ground truth back to coordinate space (cxs,cys,H,W) -> (x_min,y_min,x_max,y_max) 
        bxbybwbh_gt = self.txtytwth_2_bxbybwbh(gt_targets[...,1:5])
        x1y1x2y2_gt = exists * self.bxbybwbh_2_x1y1x2y2(bxbybwbh_gt,False)

        # Getting IOU for each prediction and zeroing non-best predictions
        iou = None
        with torch.no_grad():
            iou = self.score_iou(x1y1x2y2_gt,x1y1x2y2_preds)
            iou = iou.to(self.device)

        # Setting the IOU as the confidence predictions
        conf_preds = conf_preds.view(conf_preds.shape[0]*conf_preds.shape[1]*conf_preds.shape[2],-1).contiguous() # [B*S*S*N,1]
        iou = iou.view(iou.shape[0]*iou.shape[1],-1).contiguous()
        conf_preds[...,0] = iou[...,0]
        conf_preds = conf_preds.view(B,S*S,N,-1).contiguous()

        # Check for NaNs
        conf_preds[torch.isnan(conf_preds)] = 0

        #txtytwth_preds = self.get_best_box(txtytwth_preds,iou,x.shape[-1]//self.config['grid_stride'])

        best_box_preds = self.filter_by_best_box(txtytwth_preds,cls_preds,conf_preds,iou,S)
        conf_preds = best_box_preds[...,0:1]
        txtytwth_preds = best_box_preds[...,1:5]
        cls_preds = best_box_preds[...,5:]
        return txtytwth_preds,cls_preds,conf_preds
    
    def testing_forward(self,x,gt_targets):
        """
        Performs a forward pass in testing phase and performs 
        post processing of predictions for model. Post processing
        consists of removal of negative coordinate predictions and 
        performs non-max suppression.
        """
        # [416,416,416,416]
        scale = torch.from_numpy(np.array([[x.shape[-1], x.shape[-1], x.shape[-1], x.shape[-1]]])).to(self.device)
        # Go thru general training pass
        txtytwth_preds,cls_preds,conf_preds = self.forward_pass(x)
        B,S,N = txtytwth_preds.shape[0],int(np.sqrt(txtytwth_preds.shape[1])),txtytwth_preds.shape[2]        
        with torch.no_grad():
            bxbybwbh_preds = self.txtytwth_2_bxbybwbh(txtytwth_preds)
            x1y1x2y2_preds = self.bxbybwbh_2_x1y1x2y2(bxbybwbh_preds,False)

            x1y1x2y2_preds = x1y1x2y2_preds.view(B*S*S*N,4).contiguous()
            conf_preds = conf_preds.view(B*S*S*N,1).contiguous()
            cls_preds = cls_preds.view(B*S*S*N,self.config['num_classes']).contiguous()

            x1y1x2y2_preds = x1y1x2y2_preds / scale
            conf_preds = torch.sigmoid(conf_preds)
            cls_preds = F.softmax(cls_preds,dim=-1)

            # Attach all predictions together into a single tensor
            preds = torch.cat([conf_preds,x1y1x2y2_preds,cls_preds],dim=-1).to(self.device)

            preds = self.postprocess(preds,S)

            conf_preds = preds[...,0:1]
            bbox_preds = preds[...,1:5]
            cls_preds = preds[...,5:]

            # Scale boxes from single grid cell space to image space => [13,13,13,13]
            gc_scale = np.array([[S,S,S,S]])
            bbox_preds *= gc_scale

            return bbox_preds,cls_preds,conf_preds,preds
    
    def postprocess(self,preds,S):
        conf_thresh = self.config['conf_thresh']

        preds = preds.cpu() # send to cpu() if on gpu
        preds_ = np.zeros_like(preds)

        # -- Remove only bad conf predictions --
        preds = np.where(preds[:,0:1] >= conf_thresh,preds,preds_)

        # Only keep the positive boxes
        preds = np.where(preds[:,1:2] >= 0,preds,preds_)
        preds = np.where(preds[:,2:3] >= 0,preds,preds_)
        preds = np.where(preds[:,3:4] >= 0,preds,preds_)
        preds = np.where(preds[:,4:5] >= 0,preds,preds_)

        # Keep the boxes that are within the bounds of the image
        preds = np.where(preds[:,1:2] * (S * self.config['grid_stride']) < S * self.config['grid_stride'],preds,preds_)
        preds = np.where(preds[:,2:3] * (S * self.config['grid_stride']) < S * self.config['grid_stride'],preds,preds_)
        preds = np.where(preds[:,3:4] * (S * self.config['grid_stride']) < S * self.config['grid_stride'],preds,preds_)
        preds = np.where(preds[:,4:5] * (S * self.config['grid_stride']) < S * self.config['grid_stride'],preds,preds_)

        # Remove the zero rows
        preds = preds[np.all(preds[:,:] != 0,axis=1)]

        # Ensure that preds still has predictions in it after filtration
        if(preds.shape[0] > 0):
            # NMS for each class
            preds = self.non_max_suppression(preds,self.config['nms_thresh'],self.config['num_classes'])

        return preds
    
    def non_max_suppression(self,preds,nms_thresh,num_classes):
        filtered_preds = np.empty((0,preds.shape[1]))
        # NMS for each class
        for class_idx in range(num_classes):
            # Get the predictions for current class
            cls_locs = np.where(np.argmax(preds[:,5:],axis=1) ==  class_idx)
            curr_cls_preds = preds[cls_locs]
            nms_cls_preds = self.non_max_suppression_helper(curr_cls_preds,nms_thresh)
            if (len(nms_cls_preds) > 0):
                filtered_preds = np.concatenate((filtered_preds,nms_cls_preds),axis=0)
        return filtered_preds

    def non_max_suppression_helper(self,preds,nms_thresh):
        """
        Standard Non-Max Suppression algorithm for 
        removing boxes based on a threshold on the IOU.
        Based on Andrew Ng explanation of NMS.
        """
        def calculate_iou(best_pred,preds,areas):
            # Remove bboxes with IOU >= Thresh
            max_min_x = torch.maximum(best_pred[1],preds[1:,1])
            max_min_y = torch.maximum(best_pred[2],preds[1:,2])

            min_max_x = torch.minimum(best_pred[3],preds[1:,3])
            min_max_y = torch.minimum(best_pred[4],preds[1:,4])

            intersection_x = min_max_x - max_min_x
            intersection_y = min_max_y - max_min_y
            intersection_area = intersection_x * intersection_y

            iou = intersection_area / (areas[0] + areas[1:] - intersection_area)
            return iou
        
        def sort_confidence(preds,remove_zero_conf=False):
            # Sort by most confident boxes
            sorted_preds,sorted_pred_idxs = torch.sort(preds[...,0:1],dim=0,descending=True)
            # Use sorted idicies to sort the original predictions
            preds = preds[sorted_pred_idxs].squeeze(1)
            if(remove_zero_conf):
                # Grab only bboxes with confidence scores & remove all 0 conf bboxes
                exists_box = torch.where(sorted_preds[:,0] > 0,)[0]
                sorted_pred_idxs = sorted_pred_idxs[exists_box]
                sorted_preds = sorted_preds[exists_box]
                preds = preds[exists_box]
            areas = (preds[:,3] - preds[:,1]) * (preds[:,4] - preds[:,2])
            return preds,areas
        
        preds = torch.from_numpy(preds)
        preds,areas = sort_confidence(preds,remove_zero_conf=True)
        #print("Number of preds left = ", preds.shape)

        kept_bboxes = []
        bbox_idx = 0
        while(preds.size()[0] > 0):
            best_pred = preds[0,:]
            kept_bboxes.append(best_pred)
            iou = calculate_iou(best_pred,preds,areas)
            preds = preds[1:,:]
            # Find bboxes with IOU >= conf thresh
            non_inters = torch.where(iou < nms_thresh)[0]
            preds = preds[non_inters]
            # Re-sort by the confidence
            preds,areas = sort_confidence(preds)
        if (len(kept_bboxes) > 0):
            kept_bboxes_ = np.zeros((len(kept_bboxes),kept_bboxes[0].shape[0]))
            for idx,bbox in enumerate(kept_bboxes):
                kept_bboxes_[idx,:] = bbox
            return kept_bboxes_
        else:
            return kept_bboxes

    def forward(self,x,gt_targets):
        """
        YOLOv2 forward pass depends on the mode that the model is set to
        """
        if(self.mode == 'train'):
            txtytwth_preds,cls_preds,conf_preds = self.training_forward(x,gt_targets)
            return txtytwth_preds,cls_preds,conf_preds
        elif(self.mode == 'test'):
            txtytwth_preds,cls_preds,conf_preds,preds = self.testing_forward(x,gt_targets)
            return txtytwth_preds,cls_preds,conf_preds,preds
        
    def load_from_npz(self, weights_loc, num_conv=None):
        print("== Loading Pretrained Weights ==")
        from collections import OrderedDict
        def split(delimiters, string, maxsplit=0):
            import re
            regexPattern = '|'.join(map(re.escape, delimiters))
            return re.split(regexPattern, string, maxsplit)

        dest_src = OrderedDict(self.config['yolov2'])        
        params = np.load(weights_loc)
        own_dict = self.state_dict()
        keys = list(own_dict.keys())
        
        layer_num = 0
        for idx,(key,val) in enumerate(dest_src.items()):
            if(key.split('.',1)[0] == 'backbone'):
                key = key.split('.',1)[1]
                layer_num = int(split([".","_"],key)[0].split('v')[1])
                list_key = key.split('.')
                ptype = dest_src['backbone.{}.{}'.format(list_key[-2], list_key[-1])]
                src_key = '{}-convolutional/{}:0'.format(layer_num, ptype)
                param = torch.from_numpy(params[src_key])
                if(key == 'conv18.weight' and ptype == 'kernel'):
                    param = param[:,:,:,:4]
                    param = param.permute(3, 2, 0, 1)
                elif(ptype == 'kernel'):
                    param = param.permute(3, 2, 0, 1)
                if(key == 'conv18.bias'):
                    param = param[:4]
                own_dict['backbone.'+key].copy_(param)
            else:
                print("-->",key)
        print("== Finished Loading Pretrained Weights ==")

def main():
    import yaml
    config = yaml.safe_load(open("../config.yaml"))
    model = YOLOv2(config)
    p = model.state_dict()
    for k,v in p.items():
        print(k)
    model.load_from_npz('../weights/darknet19.weights.npz', num_conv=18)

    # m2 = YOLOv2(config)


    # n = get_n_params(model)
    # n2 = get_n_params(m2)
    # print(n)
    # print(n2)

    # m2.open_weights()

    #for layer_idx,layer in enumerate(model.named_modules()):
    #    if(isinstance(layer[1],nn.Conv2d)):
  
if __name__ == '__main__':
    main()