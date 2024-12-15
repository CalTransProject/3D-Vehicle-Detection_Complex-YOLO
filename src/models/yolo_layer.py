"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.07.05
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: This script for the yolo layer

# Refer: https://github.com/Tianxiaomo/pytorch-YOLOv4
# Refer: https://github.com/VCasecnikovs/Yet-Another-YOLOv4-Pytorch
"""

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('../')

from utils.torch_utils import to_cpu
from utils.iou_rotated_boxes_utils import iou_pred_vs_target_boxes, iou_rotated_boxes_targets_vs_anchors, \
    get_polygons_areas_fix_xy
from utils.bbox_utils import bbox_iou


class YoloLayer(nn.Module):
    """Yolo layer"""

    def __init__(self, num_classes, anchors, stride=32, scale_x_y=None, ignore_thresh=0.7, img_dim=608):
        super(YoloLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.img_dim = img_dim
        self.stride = stride
        self.scale_x_y = scale_x_y
        self.ignore_thresh = ignore_thresh
        self.grid_size = 0
        self.grid = None
        self.anchor_w = None
        self.anchor_h = None
        self.scaled_anchors = None
        self.scaled_anchors_polygons = None
        self.scaled_anchors_areas = None
        
        # Register anchors as a buffer so it moves to the correct device
        self.register_buffer('anchor_tensor', torch.tensor(anchors))
        
        # Loss coefficients
        self.lambda_coord = 1
        self.lambda_obj = 5
        self.lambda_noobj = 1
        self.lambda_cls = 1
        self.lambda_euler = 1
        
        # Define loss functions
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.bce_loss = nn.BCELoss(reduction='mean')
    
    def create_grid(self, grid_size):
        """
        Create the grid used for adjusting predictions
        Args:
            grid_size: Size of the grid (e.g. 13, 26, 52)
        Returns:
            grid: Grid coordinates tensor of shape (grid_size, grid_size)
        """
        device = self.anchor_tensor.device
        
        # Generate grid cells
        x = torch.arange(grid_size, device=device, dtype=torch.float)
        y = torch.arange(grid_size, device=device, dtype=torch.float)
        
        # Create grid using meshgrid
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        # Reshape and combine
        grid = torch.stack((xx, yy), dim=-1)
        
        return grid
    
    def compute_grid_offsets(self, grid_size):
        """
        Compute and store important grid parameters
        Args:
            grid_size: Size of the grid
        """
        self.grid_size = grid_size
        self.stride = self.img_dim / self.grid_size
        
        # Calculate offsets for each grid
        self.grid = self.create_grid(grid_size)
        
        # Calculate scaled anchors
        self.scaled_anchors = self.anchor_tensor / self.stride
        
        # Create anchor tensors
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))
        
        # Pre-compute anchor polygons and areas for rotated box IoU calculation
        self.scaled_anchors_polygons, self.scaled_anchors_areas = get_polygons_areas_fix_xy(
            self.scaled_anchors)
    
    def build_targets(self, pred_boxes, pred_cls, target, anchors):
        """ Built yolo targets to compute loss
        :param out_boxes: [num_samples or batch, num_anchors, grid_size, grid_size, 6]
        :param pred_cls: [num_samples or batch, num_anchors, grid_size, grid_size, num_classes]
        :param target: [num_boxes, 8]
        :param anchors: [num_anchors, 4]
        :return:
        """
        nB, nA, nG, _, nC = pred_cls.size()
        n_target_boxes = target.size(0)

        # Create output tensors on "device"
        obj_mask = torch.full(size=(nB, nA, nG, nG), fill_value=0, device=self.anchor_tensor.device, dtype=torch.uint8)
        noobj_mask = torch.full(size=(nB, nA, nG, nG), fill_value=1, device=self.anchor_tensor.device, dtype=torch.uint8)
        class_mask = torch.full(size=(nB, nA, nG, nG), fill_value=0, device=self.anchor_tensor.device, dtype=torch.float)
        iou_scores = torch.full(size=(nB, nA, nG, nG), fill_value=0, device=self.anchor_tensor.device, dtype=torch.float)
        tx = torch.full(size=(nB, nA, nG, nG), fill_value=0, device=self.anchor_tensor.device, dtype=torch.float)
        ty = torch.full(size=(nB, nA, nG, nG), fill_value=0, device=self.anchor_tensor.device, dtype=torch.float)
        tw = torch.full(size=(nB, nA, nG, nG), fill_value=0, device=self.anchor_tensor.device, dtype=torch.float)
        th = torch.full(size=(nB, nA, nG, nG), fill_value=0, device=self.anchor_tensor.device, dtype=torch.float)
        tim = torch.full(size=(nB, nA, nG, nG), fill_value=0, device=self.anchor_tensor.device, dtype=torch.float)
        tre = torch.full(size=(nB, nA, nG, nG), fill_value=0, device=self.anchor_tensor.device, dtype=torch.float)
        tcls = torch.full(size=(nB, nA, nG, nG, nC), fill_value=0, device=self.anchor_tensor.device, dtype=torch.float)
        tconf = obj_mask.float()
        giou_loss = torch.tensor([0.], device=self.anchor_tensor.device, dtype=torch.float)

        if n_target_boxes > 0:  # Make sure that there is at least 1 box
            try:
                b, target_labels = target[:, :2].long().t()
                target_boxes = torch.cat((target[:, 2:6] * nG, target[:, 6:8]), dim=-1)  # scale up x, y, w, h

                gxy = target_boxes[:, :2]
                gwh = target_boxes[:, 2:4]
                gimre = target_boxes[:, 4:6]

                gx, gy = gxy.t()
                gw, gh = gwh.t()
                gim, gre = gimre.t()

                # Get grid box indices
                gi = gx.long()
                gj = gy.long()

                # Get shape of gt box
                gt_boxes = torch.zeros((n_target_boxes, 6), device=self.anchor_tensor.device)
                gt_boxes[:, :2] = gxy
                gt_boxes[:, 2:4] = gwh
                gt_boxes[:, 4:] = gimre

                # Get shape of anchor boxes
                anchor_shapes = torch.zeros((self.num_anchors, 4), device=self.anchor_tensor.device)
                anchor_shapes[:, 2:4] = anchors  # width, height

                # Calculate iou between gt and anchor shapes
                anch_ious = torch.zeros((n_target_boxes, self.num_anchors), device=self.anchor_tensor.device)
                for i in range(n_target_boxes):
                    for j in range(self.num_anchors):
                        anch_ious[i, j] = bbox_iou(gt_boxes[i, :4], anchor_shapes[j], x1y1x2y2=False)

                # For each target, find the best matching anchor
                best_n = anch_ious.max(1)[1]

                # Set masks
                for i in range(n_target_boxes):
                    obj_mask[b[i], best_n[i], gj[i], gi[i]] = 1
                    noobj_mask[b[i], best_n[i], gj[i], gi[i]] = 0

                    # Coordinates
                    tx[b[i], best_n[i], gj[i], gi[i]] = gx[i] - gi[i].float()
                    ty[b[i], best_n[i], gj[i], gi[i]] = gy[i] - gj[i].float()

                    # Width and height
                    tw[b[i], best_n[i], gj[i], gi[i]] = torch.log(gw[i] / anchors[best_n[i]][0] + 1e-16)
                    th[b[i], best_n[i], gj[i], gi[i]] = torch.log(gh[i] / anchors[best_n[i]][1] + 1e-16)

                    # Im and Re
                    tim[b[i], best_n[i], gj[i], gi[i]] = gim[i]
                    tre[b[i], best_n[i], gj[i], gi[i]] = gre[i]

                    # One-hot encoding of label
                    tcls[b[i], best_n[i], gj[i], gi[i], target_labels[i]] = 1

                    # Compute label correctness and iou at best anchor
                    class_mask[b[i], best_n[i], gj[i], gi[i]] = (pred_cls[b[i], best_n[i], gj[i], gi[i]].argmax(-1) == target_labels[i]).float()
                    
                    # Calculate IoU scores for this target
                    pred_box = pred_boxes[b[i], best_n[i], gj[i], gi[i]]
                    target_box = gt_boxes[i]
                    iou_scores[b[i], best_n[i], gj[i], gi[i]] = bbox_iou(pred_box[:4], target_box[:4], x1y1x2y2=False)

                    if self.use_giou_loss:
                        giou = bbox_iou(pred_box[:4], target_box[:4], x1y1x2y2=False, GIoU=True)
                        # Initialize giou_loss if it doesn't exist
                        if 'giou_loss' not in locals():
                            giou_loss = torch.tensor(0.0, device=self.anchor_tensor.device).view(1)
                        # Accumulate giou loss
                        giou_loss = giou_loss + (1 - giou)

                tconf = obj_mask.float()
            except Exception as e:
                print(f"\nError in build_targets: {str(e)}")
                print(f"target shape: {target.shape}")
                print(f"target contents: {target}")
                print(f"pred_boxes shape: {pred_boxes.shape}")
                print(f"pred_cls shape: {pred_cls.shape}")
                print(f"anchors shape: {anchors.shape}")
                raise e
        else:
            giou_loss = torch.tensor(0, device=self.anchor_tensor.device).view(1)

        return iou_scores, giou_loss, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tim, tre, tcls, tconf

    def forward(self, x, targets=None, img_size=608, use_giou_loss=False):
        """
        :param x: [batch_size, num_anchors * (6 + 1 + num_classes), grid_size, grid_size]
        :param targets: [num_boxes, 8] (box_idx, class, x, y, w, l, sin(yaw), cos(yaw))
        :param img_size: default 608
        :return:
        """
        self.img_size = img_size
        self.use_giou_loss = use_giou_loss

        num_samples = x.size(0)
        grid_size = x.size(2)

        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + 7, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        im = prediction[..., 4]  # angle imaginary part
        re = prediction[..., 5]  # angle real part
        pred_conf = torch.sigmoid(prediction[..., 6])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 7:])  # Cls pred.

        try:
            # Add offset and scale with anchors
            pred_boxes = x.new(prediction[..., :6].shape)
            self.compute_grid_offsets(grid_size)
            grid = self.grid.repeat(num_samples, self.num_anchors, 1, 1, 1)
            
            pred_boxes[..., 0] = (x + grid[..., 0]) / grid_size
            pred_boxes[..., 1] = (y + grid[..., 1]) / grid_size
            pred_boxes[..., 2] = (w.exp() * self.anchor_w) / grid_size
            pred_boxes[..., 3] = (h.exp() * self.anchor_h) / grid_size
            pred_boxes[..., 4] = im
            pred_boxes[..., 5] = re

            output = torch.cat((
                pred_boxes[..., :6].view(num_samples, -1, 6),
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ), -1)

            if targets is not None:
                iou_scores, giou_loss, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tim, tre, tcls, tconf = self.build_targets(
                    pred_boxes=pred_boxes, pred_cls=pred_cls, target=targets, anchors=self.scaled_anchors
                )

                # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
                loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
                loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
                loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
                loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
                loss_im = self.mse_loss(im[obj_mask], tim[obj_mask])
                loss_re = self.mse_loss(re[obj_mask], tre[obj_mask])
                loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
                loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
                loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
                loss_conf = loss_conf_obj + loss_conf_noobj

                total_loss = loss_x + loss_y + loss_w + loss_h + loss_im + loss_re + loss_conf + loss_cls
                if self.use_giou_loss and giou_loss is not None:
                    # Ensure giou_loss has the correct shape and type
                    if not isinstance(giou_loss, torch.Tensor):
                        giou_loss = torch.tensor(giou_loss, device=total_loss.device)
                    if giou_loss.dim() == 0:
                        giou_loss = giou_loss.view(1)
                    # Scale giou_loss to match other losses
                    giou_loss = giou_loss.mean()  # Average across all targets
                    total_loss += giou_loss

                # Metrics (store loss values for logging)
                cls_acc = 100 * class_mask[obj_mask].mean()
                conf_obj = pred_conf[obj_mask].mean()
                conf_noobj = pred_conf[noobj_mask].mean()
                conf50 = (pred_conf > 0.5).float()
                iou50 = (iou_scores > 0.5).float()
                iou75 = (iou_scores > 0.75).float()
                detected_mask = conf50 * class_mask * tconf
                precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
                recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
                recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

                self.metrics = {
                    "loss": to_cpu(total_loss).item(),
                    "x": to_cpu(loss_x).item(),
                    "y": to_cpu(loss_y).item(),
                    "w": to_cpu(loss_w).item(),
                    "h": to_cpu(loss_h).item(),
                    "im": to_cpu(loss_im).item(),
                    "re": to_cpu(loss_re).item(),
                    "conf": to_cpu(loss_conf).item(),
                    "cls": to_cpu(loss_cls).item(),
                    "cls_acc": to_cpu(cls_acc).item(),
                    "recall50": to_cpu(recall50).item(),
                    "recall75": to_cpu(recall75).item(),
                    "precision": to_cpu(precision).item(),
                    "conf_obj": to_cpu(conf_obj).item(),
                    "conf_noobj": to_cpu(conf_noobj).item(),
                    "grid_size": grid_size,
                }

                return output, total_loss
            
        except Exception as e:
            print(f"\nError in YoloLayer forward pass: {str(e)}")
            print(f"Input tensor x shape: {x.shape}")
            print(f"Prediction tensor shape: {prediction.shape}")
            if targets is not None:
                print(f"Targets shape: {targets.shape}")
                print(f"Targets contents: {targets}")
                print(f"Grid size: {grid_size}")
                print(f"Number of samples: {num_samples}")
            raise e

        return output, 0
