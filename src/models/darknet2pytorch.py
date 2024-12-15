"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Refer: https://github.com/Tianxiaomo/pytorch-YOLOv4
"""

import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as  np

sys.path.append('../')

from models.yolo_layer import YoloLayer
from models.darknet_utils import parse_cfg, print_cfg, load_fc, load_conv_bn, load_conv
from utils.torch_utils import to_cpu


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


class MaxPoolDark(nn.Module):
    def __init__(self, size=2, stride=1):
        super(MaxPoolDark, self).__init__()
        self.size = size
        self.stride = stride

    def forward(self, x):
        '''
        darknet output_size = (input_size + p - k) / s +1
        p : padding = k - 1
        k : size
        s : stride
        torch output_size = (input_size + 2*p -k) / s +1
        p : padding = k//2
        '''
        p = self.size // 2
        if ((x.shape[2] - 1) // self.stride) != ((x.shape[2] + 2 * p - self.size) // self.stride):
            padding1 = (self.size - 1) // 2
            padding2 = padding1 + 1
        else:
            padding1 = (self.size - 1) // 2
            padding2 = padding1
        if ((x.shape[3] - 1) // self.stride) != ((x.shape[3] + 2 * p - self.size) // self.stride):
            padding3 = (self.size - 1) // 2
            padding4 = padding3 + 1
        else:
            padding3 = (self.size - 1) // 2
            padding4 = padding3
        x = F.max_pool2d(F.pad(x, (padding3, padding4, padding1, padding2), mode='replicate'),
                         self.size, stride=self.stride)
        return x


class Upsample_expand(nn.Module):
    def __init__(self, stride=2):
        super(Upsample_expand, self).__init__()
        self.stride = stride

    def forward(self, x):
        stride = self.stride
        assert (x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        ws = stride
        hs = stride
        x = x.view(B, C, H, 1, W, 1).expand(B, C, H, stride, W, stride).contiguous().view(B, C, H * stride, W * stride)
        return x


class Upsample_interpolate(nn.Module):
    def __init__(self, stride):
        super(Upsample_interpolate, self).__init__()
        self.stride = stride

    def forward(self, x):
        x_numpy = x.cpu().detach().numpy()
        H = x_numpy.shape[2]
        W = x_numpy.shape[3]

        H = H * self.stride
        W = W * self.stride

        out = F.interpolate(x, size=(H, W), mode='nearest')
        return out


class Reorg(nn.Module):
    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride

    def forward(self, x):
        stride = self.stride
        assert (x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        assert (H % stride == 0)
        assert (W % stride == 0)
        ws = stride
        hs = stride
        x = x.view(B, C, H / hs, hs, W / ws, ws).transpose(3, 4).contiguous()
        x = x.view(B, C, H / hs * W / ws, hs * ws).transpose(2, 3).contiguous()
        x = x.view(B, hs * ws * C, H / hs, W / ws).transpose(1, 2).contiguous()
        x = x.view(B, hs * ws * C, H / hs, W / ws)
        return x


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        N = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        x = F.avg_pool2d(x, (H, W))
        x = x.view(N, C)
        return x


# for route and shortcut
class EmptyModule(nn.Module):
    def __init__(self):
        super(EmptyModule, self).__init__()

    def forward(self, x):
        return x


# support route shortcut and reorg
class Darknet(nn.Module):
    def __init__(self, cfgfile, use_giou_loss):
        super(Darknet, self).__init__()
        self.use_giou_loss = use_giou_loss
        self.blocks = parse_cfg(cfgfile)
        self.width = int(self.blocks[0]['width'])
        self.height = int(self.blocks[0]['height'])

        self.models = self.create_network(self.blocks)  # merge conv, bn,leaky
        self.yolo_layers = [layer for layer in self.models if layer.__class__.__name__ == 'YoloLayer']

        self.loss = self.models[len(self.models) - 1]

        self.header = torch.IntTensor([0, 0, 0, 0])
        self.seen = 0

    def forward(self, x, targets=None):
        # batch_size, c, h, w
        img_size = x.size(2)
        ind = -2
        self.loss = None
        outputs = dict()
        loss = 0.
        yolo_outputs = []
        layer_outputs = []
        output = []
        
        # Validate input channels
        if x.size(1) != int(self.blocks[0]['channels']):
            raise ValueError(f"Expected input to have {self.blocks[0]['channels']} channels, but got {x.size(1)} channels")
            
        print(f"\nInput tensor shape: {x.shape}")
        outputs[-1] = x  # Store input tensor
        for i, block in enumerate(self.blocks[1:]):
            ind = i + 1
            if block['type'] == 'net':
                continue
            elif block['type'] in ['convolutional', 'maxpool', 'reorg', 'upsample', 'avgpool', 'softmax', 'connected']:
                print(f"\nProcessing layer {ind} ({block['type']})")
                print(f"Input shape: {x.shape}")
                x = self.models[ind](x)
                print(f"Output shape: {x.shape}")
                outputs[ind] = x
            elif block['type'] == 'route':
                print(f"\nProcessing route layer at index {ind}")
                layers = block['layers'].split(',')
                layers = [int(i) if int(i) > 0 else int(i) + ind for i in layers]
                print(f"Route connections from layers: {layers}")
                
                if len(layers) == 1:
                    if 'groups' not in block.keys() or int(block['groups']) == 1:
                        x = outputs[layers[0]]
                        print(f"Single route from layer {layers[0]}, tensor shape: {x.shape}")
                        if 'channels' in block:
                            expected_channels = int(block['channels'])
                            if expected_channels != x.size(1):
                                raise ValueError(f"Route layer expected {expected_channels} channels, but got {x.size(1)}")
                        outputs[ind] = x
                    else:
                        groups = int(block['groups'])
                        group_id = int(block['group_id'])
                        _, b, _, _ = outputs[layers[0]].shape
                        x = outputs[layers[0]][:, b // groups * group_id:b // groups * (group_id + 1)]
                        print(f"Grouped route, tensor shape after grouping: {x.shape}")
                        outputs[ind] = x
                elif len(layers) == 2:
                    x1 = outputs[layers[0]]
                    x2 = outputs[layers[1]]
                    print(f"Route concatenating from:")
                    print(f"  Layer {layers[0]}: shape={x1.shape}")
                    print(f"  Layer {layers[1]}: shape={x2.shape}")
                    
                    # Add spatial dimension check and adjustment
                    if x1.shape[2:] != x2.shape[2:]:
                        # Downsample the tensor with larger spatial dimensions
                        if x1.shape[2] > x2.shape[2]:
                            x1 = F.interpolate(x1, size=x2.shape[2:], mode='nearest')
                        else:
                            x2 = F.interpolate(x2, size=x1.shape[2:], mode='nearest')
                        print(f"After resizing:")
                        print(f"  Layer {layers[0]}: shape={x1.shape}")
                        print(f"  Layer {layers[1]}: shape={x2.shape}")
                    
                    # Verify channel count if specified
                    if 'channels' in block:
                        expected_channels = int(block['channels'])
                        actual_channels = x1.size(1) + x2.size(1)
                        if expected_channels != actual_channels:
                            raise ValueError(f"Route layer expected {expected_channels} channels after concatenation, but got {actual_channels}")
                    
                    x = torch.cat((x1, x2), 1)
                    print(f"After concatenation: shape={x.shape}")
                    outputs[ind] = x
                elif len(layers) == 4:
                    x1 = outputs[layers[0]]
                    x2 = outputs[layers[1]]
                    x3 = outputs[layers[2]]
                    x4 = outputs[layers[3]]
                    x = torch.cat((x1, x2, x3, x4), 1)
                    outputs[ind] = x
                else:
                    print("rounte number > 2 ,is {}".format(len(layers)))

            elif block['type'] == 'shortcut':
                from_layer = int(block['from'])
                activation = block['activation']
                from_layer = from_layer if from_layer > 0 else from_layer + ind
                x1 = outputs[from_layer]
                x2 = outputs[ind - 1]
                x = x1 + x2
                if activation == 'leaky':
                    x = F.leaky_relu(x, 0.1, inplace=True)
                elif activation == 'relu':
                    x = F.relu(x, inplace=True)
                outputs[ind] = x
            elif block['type'] == 'yolo':
                x, layer_loss = self.models[ind](x, targets, img_size, self.use_giou_loss)
                loss += layer_loss
                yolo_outputs.append(x)
            elif block['type'] == 'cost':
                continue
            else:
                print('unknown type %s' % (block['type']))
        yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))

        return yolo_outputs if targets is None else (loss, yolo_outputs)

    def print_network(self):
        print_cfg(self.blocks)

    def create_network(self, blocks):
        models = nn.ModuleList()

        # Initialize with the number of input channels from the [net] block
        input_channels = int(blocks[0]['channels'])
        prev_filters = input_channels  # Start with input channels
        out_filters = []
        prev_stride = 1
        out_strides = []
        conv_id = 0
        ind = 0  # Add index counter

        # Add an empty module at index 0 to maintain indexing
        models.append(EmptyModule())
        out_filters.append(input_channels)
        out_strides.append(prev_stride)
        ind += 1

        for block in blocks[1:]:  # Skip the [net] block
            if block['type'] == 'convolutional':
                conv_id = conv_id + 1
                batch_normalize = int(block['batch_normalize'])
                filters = int(block['filters'])
                kernel_size = int(block['size'])
                stride = int(block['stride'])
                is_pad = int(block['pad'])
                pad = (kernel_size - 1) // 2 if is_pad else 0
                activation = block['activation']
                
                # Use explicit in_channels if specified, otherwise use prev_filters
                in_channels = int(block['in_channels']) if 'in_channels' in block else prev_filters
                
                print(f"\nCreating conv layer {conv_id}:")
                print(f"  in_channels: {in_channels}")
                print(f"  out_channels: {filters}")
                print(f"  kernel_size: {kernel_size}")
                print(f"  stride: {stride}")
                print(f"  padding: {pad}")
                
                model = nn.Sequential()
                if batch_normalize:
                    model.add_module('conv{0}'.format(conv_id),
                                   nn.Conv2d(in_channels, filters, kernel_size, stride, pad, bias=False))
                    model.add_module('bn{0}'.format(conv_id), nn.BatchNorm2d(filters))
                else:
                    model.add_module('conv{0}'.format(conv_id),
                                   nn.Conv2d(in_channels, filters, kernel_size, stride, pad))
                if activation == 'leaky':
                    model.add_module('leaky{0}'.format(conv_id), nn.LeakyReLU(0.1, inplace=True))
                elif activation == 'relu':
                    model.add_module('relu{0}'.format(conv_id), nn.ReLU(inplace=True))
                elif activation == 'mish':
                    model.add_module('mish{0}'.format(conv_id), Mish())
                else:
                    print("[INFO] No error, the convolution haven't activate {}".format(activation))

                prev_filters = filters
                out_filters.append(prev_filters)
                prev_stride = stride * prev_stride
                out_strides.append(prev_stride)
                models.append(model)
                ind += 1

            elif block['type'] == 'maxpool':
                pool_size = int(block['size'])
                stride = int(block['stride'])
                if stride == 1 and pool_size % 2:
                    # You can use Maxpooldark instead, here is convenient to convert
                    model = nn.MaxPool2d(kernel_size=pool_size, stride=stride, padding=pool_size // 2)
                elif stride == pool_size:
                    model = nn.MaxPool2d(kernel_size=pool_size, stride=stride, padding=0)
                else:
                    model = MaxPoolDark(pool_size, stride)
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(model)
                ind += 1

            elif block['type'] == 'avgpool':
                model = GlobalAvgPool2d()
                out_filters.append(prev_filters)
                models.append(model)
                ind += 1

            elif block['type'] == 'softmax':
                model = nn.Softmax()
                out_strides.append(prev_stride)
                out_filters.append(prev_filters)
                models.append(model)
                ind += 1

            elif block['type'] == 'cost':
                if block['_type'] == 'sse':
                    model = nn.MSELoss(reduction='mean')
                elif block['_type'] == 'L1':
                    model = nn.L1Loss(reduction='mean')
                elif block['_type'] == 'smooth':
                    model = nn.SmoothL1Loss(reduction='mean')
                out_filters.append(1)
                out_strides.append(prev_stride)
                models.append(model)
                ind += 1

            elif block['type'] == 'reorg':
                stride = int(block['stride'])
                prev_filters = stride * stride * prev_filters
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(Reorg(stride))
                ind += 1

            elif block['type'] == 'upsample':
                stride = int(block['stride'])
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(Upsample_expand(stride))
                ind += 1

            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                ind_from = [int(i) if int(i) > 0 else int(i) + ind for i in layers]
                if len(ind_from) == 1:
                    if 'groups' not in block.keys() or int(block['groups']) == 1:
                        prev_filters = out_filters[ind_from[0]]
                        prev_stride = out_strides[ind_from[0]]
                    else:
                        prev_filters = out_filters[ind_from[0]] // int(block['groups'])
                        prev_stride = out_strides[ind_from[0]] // int(block['groups'])
                elif len(ind_from) == 2:
                    assert (ind_from[0] == ind - 1 or ind_from[1] == ind - 1)
                    if 'channels' in block:
                        prev_filters = int(block['channels'])
                    else:
                        prev_filters = out_filters[ind_from[0]] + out_filters[ind_from[1]]
                    prev_stride = out_strides[ind_from[0]]
                elif len(ind_from) == 4:
                    assert (ind_from[0] == ind - 1)
                    if 'channels' in block:
                        prev_filters = int(block['channels'])
                    else:
                        prev_filters = out_filters[ind_from[0]] + out_filters[ind_from[1]] + out_filters[ind_from[2]] + out_filters[ind_from[3]]
                    prev_stride = out_strides[ind_from[0]]
                else:
                    print("route error!!!")

                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(EmptyModule())
                ind += 1

            elif block['type'] == 'shortcut':
                ind_from = int(block['from'])
                ind_from = ind_from if ind_from > 0 else ind_from + ind
                out_filters.append(out_filters[ind_from])
                out_strides.append(out_strides[ind_from])
                models.append(EmptyModule())
                ind += 1

            elif block['type'] == 'connected':
                filters = int(block['output'])
                if block['activation'] == 'linear':
                    model = nn.Linear(prev_filters, filters)
                elif block['activation'] == 'leaky':
                    model = nn.Sequential(
                        nn.Linear(prev_filters, filters),
                        nn.LeakyReLU(0.1, inplace=True))
                elif block['activation'] == 'relu':
                    model = nn.Sequential(
                        nn.Linear(prev_filters, filters),
                        nn.ReLU(inplace=True))
                prev_filters = filters
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(model)
                ind += 1

            elif block['type'] == 'yolo':
                anchor_masks = [int(i) for i in block['mask'].split(',')]
                anchors = [float(i) for i in block['anchors'].split(',')]
                anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
                anchors = [anchors[i] for i in anchor_masks]
                num_classes = int(block['classes'])
                scale_x_y = float(block['scale_x_y']) if 'scale_x_y' in block else None
                ignore_thresh = float(block['ignore_thresh'])
                model = YoloLayer(num_classes, anchors, stride=prev_stride, scale_x_y=scale_x_y, ignore_thresh=ignore_thresh)
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(model)
                ind += 1

            else:
                print('unknown type %s' % (block['type']))

        return models

    def load_weights(self, weightfile):
        fp = open(weightfile, 'rb')
        header = np.fromfile(fp, count=5, dtype=np.int32)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        buf = np.fromfile(fp, dtype=np.float32)
        fp.close()

        start = 0
        ind = -2
        for block in self.blocks:
            if start >= buf.size:
                break
            ind = ind + 1
            if block['type'] == 'net':
                continue
            elif block['type'] == 'convolutional':
                model = self.models[ind]
                batch_normalize = int(block['batch_normalize'])
                if batch_normalize:
                    start = load_conv_bn(buf, start, model[0], model[1])
                else:
                    start = load_conv(buf, start, model[0])
            elif block['type'] == 'connected':
                model = self.models[ind]
                if block['activation'] != 'linear':
                    start = load_fc(buf, start, model[0])
                else:
                    start = load_fc(buf, start, model)
            elif block['type'] == 'maxpool':
                pass
            elif block['type'] == 'reorg':
                pass
            elif block['type'] == 'upsample':
                pass
            elif block['type'] == 'route':
                pass
            elif block['type'] == 'shortcut':
                pass
            elif block['type'] == 'yolo':
                pass
            elif block['type'] == 'avgpool':
                pass
            elif block['type'] == 'softmax':
                pass
            elif block['type'] == 'cost':
                pass
            else:
                print('unknown type %s' % (block['type']))
