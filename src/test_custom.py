"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.07.08
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: Testing script
"""

import argparse
import sys
import os
import time

from easydict import EasyDict as edict
import cv2
import torch
import numpy as np

sys.path.append('../')

import config.custom_config as cnf
from data_process import custom_data_utils, custom_bev_utils
from data_process.custom_dataloader import create_test_dataloader
from models.model_utils import create_model
from utils.misc import make_folder
# from utils.evaluation_utils import post_processing, rescale_boxes, post_processing_v2
from utils.evaluation_utils_custom import post_processing, rescale_boxes, post_processing_v2
from utils.misc import time_synchronized
# from utils.visualization_utils import show_image_with_boxes, merge_rgb_to_bev, predictions_to_kitti_format
from utils.visualization_utils_custom import show_image_with_boxes, merge_rgb_to_bev, predictions_to_custom_format


def parse_test_configs():
    parser = argparse.ArgumentParser(description='Demonstration config for Complex YOLO Implementation')
    parser.add_argument('--saved_fn', type=str, default='complexer_yolov4', metavar='FN',
                        help='The name using for saving logs, models,...')
    parser.add_argument('-a', '--arch', type=str, default='darknet', metavar='ARCH',
                        help='The name of the model architecture')
    parser.add_argument('--cfgfile', type=str, default='./config/cfg/complex_yolov4.cfg', metavar='PATH',
                        help='The path for cfgfile (only for darknet)')
    parser.add_argument('--pretrained_path', type=str, default=None, metavar='PATH',
                        help='the path of the pretrained checkpoint')
    parser.add_argument('--use_giou_loss', action='store_true',
                        help='If true, use GIoU loss during training. If false, use MSE loss for training')

    parser.add_argument('--no_cuda', action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--gpu_idx', default=None, type=int,
                        help='GPU index to use.')

    parser.add_argument('--img_size', type=int, default=608,
                        help='the size of input image')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='mini-batch size (default: 4)')

    parser.add_argument('--conf_thresh', type=float, default=0.5,
                        help='the threshold for conf')
    parser.add_argument('--nms_thresh', type=float, default=0.5,
                        help='the threshold for conf')

    parser.add_argument('--show_image', action='store_true',
                        help='If true, show the image during demostration')
    parser.add_argument('--save_test_output', action='store_true',
                        help='If true, the output image of the testing phase will be saved')
    parser.add_argument('--output_format', type=str, default='image', metavar='PATH',
                        help='the type of the test output (support image or video)')
    parser.add_argument('--output_video_fn', type=str, default='out_complexer_yolov4', metavar='PATH',
                        help='the video filename if the output format is video')

    configs = edict(vars(parser.parse_args()))
    configs.pin_memory = True

    ####################################################################
    ##############Dataset, Checkpoints, and results dir configs#########
    ####################################################################
    configs.working_dir = '../'
    configs.dataset_dir = os.path.join(configs.working_dir, 'dataset', 'custom')

    if configs.save_test_output:
        configs.results_dir = os.path.join(configs.working_dir, 'results', configs.saved_fn)
        make_folder(configs.results_dir)

    return configs


if __name__ == '__main__':
    configs = parse_test_configs()
    configs.distributed = False  # For testing

    model = create_model(configs)
    model.print_network()
    print('\n\n' + '-*=' * 30 + '\n\n')

    # device_string = 'cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx)
    device_string = 'cpu'  # Force the code to run on CPU

    assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)
    model.load_state_dict(torch.load(configs.pretrained_path, map_location=device_string))

    configs.device = torch.device(device_string)
    model = model.to(device=configs.device)

    out_cap = None

    model.eval()

    start_frame_index = 100  # Set this to your desired starting frame index
    # start_frame_index = 230  # Set this to your desired starting frame index
    test_dataloader = create_test_dataloader(configs)
    with torch.no_grad():
        for batch_idx, (img_paths, imgs_bev) in enumerate(test_dataloader):

            # Skip frames until you reach the starting index
            if batch_idx < start_frame_index:
                continue

            input_imgs = imgs_bev.to(device=configs.device).float()
            t1 = time_synchronized()
            outputs = model(input_imgs)
            t2 = time_synchronized()

            # Print the nms_thresh value for the current iteration
            print(f"Iteration {batch_idx}: nms_thresh = {configs.nms_thresh}")

            detections = post_processing_v2(outputs, conf_thresh=configs.conf_thresh, nms_thresh=configs.nms_thresh)

            img_detections = []  # Stores detections for each image index
            img_detections.extend(detections)

            img_bev = imgs_bev.squeeze() * 255
            img_bev = img_bev.permute(1, 2, 0).numpy().astype(np.uint8)
            img_bev = cv2.resize(img_bev, (configs.img_size, configs.img_size))
            for detections in img_detections:
                if detections is None:
                    continue
                # Rescale boxes to original image
                # print(detections)
                detections = rescale_boxes(detections, configs.img_size, img_bev.shape[:2])
                for x, y, w, l, im, re, *_, cls_pred in detections:

                    # Counter Clockwise Orientation
                    yaw = np.arctan2(im, re)

                    # Clockwise Orientation
                    # yaw = (-np.arctan2(im, re)) % (2 * np.pi)

                    # yaw = np.arctan2(im, re) + np.pi / 2 # -- Jonathan C. 03/27/2024
                    # yaw = np.deg2rad(yaw) # -- Jonathan C. 03/24/2024
                    # yaw = np.rad2deg(yaw)  # -- Jonathan C. 03/24/2024 back to degrees!
                    # yaw = -yaw  # -- Jonathan C. 03/27/2024

                    # Draw rotated box
                    # print("x:", x)
                    # print("y:", y)
                    # print("w:", w)
                    # print("l:", l)
                    # print("yaw:", yaw)
                    # Output bounding box coordinates and yaw
                    print(f"Bounding Box Coordinates and Orientation:")
                    print(f"Class: {cls_pred}, X: {x}, Y: {y}, Width: {w}, Length: {l}, Yaw: {yaw}")
                    custom_bev_utils.drawRotatedBox(img_bev, x, y, w, l, yaw, cnf.colors[int(cls_pred)])


            img_rgb = cv2.imread(img_paths[0])
            calib = custom_data_utils.Calibration(img_paths[0].replace(".png", ".txt").replace("image_2", "calib"))
            # objects_pred = predictions_to_kitti_format(img_detections, calib, img_rgb.shape, configs.img_size)
            objects_pred = predictions_to_custom_format(img_detections, calib, img_rgb.shape, configs.img_size)
            # print(objects_pred)
            img_rgb = show_image_with_boxes(img_rgb, objects_pred, calib, False)

            img_bev = cv2.flip(cv2.flip(img_bev, 0), 1)

            out_img = merge_rgb_to_bev(img_rgb, img_bev, output_width=608)

            print('\tDone testing the {}th sample, time: {:.1f}ms, speed {:.2f}FPS'.format(batch_idx, (t2 - t1) * 1000,
                                                                                           1 / (t2 - t1)))

            if configs.save_test_output:
                if configs.output_format == 'image':
                    img_fn = os.path.basename(img_paths[0])[:-4]
                    cv2.imwrite(os.path.join(configs.results_dir, '{}.jpg'.format(img_fn)), out_img)
                elif configs.output_format == 'video':
                    if out_cap is None:
                        out_cap_h, out_cap_w = out_img.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                        out_cap = cv2.VideoWriter(
                            os.path.join(configs.results_dir, '{}.avi'.format(configs.output_video_fn)),
                            fourcc, 30, (out_cap_w, out_cap_h))

                    out_cap.write(out_img)
                else:
                    raise TypeError

            # if configs.show_image:
            #     cv2.imshow('test-img', out_img)
            #     print('\n[INFO] Press n to see the next sample >>> Press Esc to quit...\n')
            #     if cv2.waitKey(0) & 0xFF == 27:
            #         break
            import matplotlib.pyplot as plt

            if configs.show_image:
                plt.figure(figsize=(10, 10))
                out_img_rgb = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
                plt.imshow(out_img_rgb)
                # plt.axis('off')  # Hide axes
                # Adding grid, labels, and title for better spatial understanding
                # Debugging -----------------------
                # plt.grid(which='both', color='gray', linestyle='--', linewidth=0.5)
                # plt.minorticks_on()
                # plt.xlabel('X axis')
                # plt.ylabel('Y axis')
                # plt.title('BEV Image with Bounding Boxes')
                # Debugging -----------------------
                plt.show()
                print('\n[INFO] Close the image window to see the next sample...\n')

                # if configs.show_image:
                #     plt.figure(figsize=(10, 10))
                #     img_bev_rgb = cv2.cvtColor(img_bev, cv2.COLOR_BGR2RGB)  # Convert BEV image from BGR to RGB
                #     plt.imshow(img_bev_rgb)
                # plt.imshow(img_bev_rgb, origin='lower')
                # plt.grid(which='both', color='gray', linestyle='--', linewidth=0.5)
                # plt.minorticks_on()
                # plt.xlabel('X axis')
                # plt.ylabel('Y axis')
                # plt.title('BEV Image with Bounding Boxes')

                # Move the X-axis to the top
                # ax = plt.gca()  # Get the current Axes instance on the current figure matching the given keyword args, or create one.
                # ax.xaxis.tick_top()
                # ax.xaxis.set_label_position('top')  # Move the x-axis label to the top

                # ax = plt.gca()  # Get the current Axes instance on the current figure
                # ax.invert_yaxis()  # Invert the Y-axis to start from 0 at the bottom withou

                plt.show()
                print('\n[INFO] Close the image window to see the next sample...\n')

    if out_cap:
        out_cap.release()
    # cv2.destroyAllWindows()
