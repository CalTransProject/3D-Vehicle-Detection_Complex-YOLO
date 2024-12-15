import time
import numpy as np
import sys
import random
import os
import warnings
import gc

warnings.filterwarnings("ignore", category=UserWarning)

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from tqdm import tqdm

sys.path.append('./')

from data_process.custom_dataloader import create_train_dataloader, create_val_dataloader
# from models.model_utils_custom_cpu import create_model, make_data_parallel, get_num_parameters
from models.model_utils_custom_cpu import create_model, get_num_parameters
from utils.train_utils import create_optimizer, create_lr_scheduler, get_saved_state, save_checkpoint
from utils.train_utils import reduce_tensor, to_python_float, get_tensorboard_log
from utils.misc import AverageMeter, ProgressMeter
from utils.logger import Logger
from config.train_config_custom import parse_train_configs
from evaluate_custom_cpu import evaluate_mAP


def main():
    configs = parse_train_configs()

    # Re-produce results
    if configs.seed is not None:
        random.seed(configs.seed)
        np.random.seed(configs.seed)
        torch.manual_seed(configs.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if configs.gpu_idx is not None:
        print('You have chosen a specific GPU. This will completely disable data parallelism.')

    if configs.dist_url == "env://" and configs.world_size == -1:
        configs.world_size = int(os.environ["WORLD_SIZE"])

    configs.distributed = configs.world_size > 1 or configs.multiprocessing_distributed

    if configs.multiprocessing_distributed:
        configs.world_size = configs.ngpus_per_node * configs.world_size
        mp.spawn(main_worker, nprocs=configs.ngpus_per_node, args=(configs,))
    else:
        main_worker(configs.gpu_idx, configs)


def main_worker(gpu_idx, configs):
    configs.gpu_idx = gpu_idx

    # Check if MPS is available
    if torch.backends.mps.is_available():
        configs.device = torch.device("cpu")  # Force CPU for now due to MPS compatibility issues
        print("MPS available but using CPU for better compatibility")
    else:
        configs.device = torch.device("cpu")
        print("MPS not available, using CPU")

    configs.is_master_node = True  # Since we're not using distributed training

    if configs.is_master_node:
        logger = Logger(configs.logs_dir, configs.saved_fn)
        logger.info('>>> Created a new logger')
        logger.info('>>> configs: {}'.format(configs))
        tb_writer = SummaryWriter(log_dir=os.path.join(configs.logs_dir, 'tensorboard'))
    else:
        logger = None
        tb_writer = None

    # model
    model = create_model(configs)
    model = model.to(configs.device)

    # Initialize scaler for MPS device
    if configs.device.type == 'mps':
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    # load weight from a checkpoint
    if configs.pretrained_path and configs.pretrained_path.lower() != 'none':
        assert os.path.isfile(configs.pretrained_path), "=> no checkpoint found at '{}'".format(configs.pretrained_path)
        pretrained_dict = torch.load(configs.pretrained_path, map_location=configs.device)
        model_dict = model.state_dict()
        # Filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == pretrained_dict[k].shape}
        # Overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        if logger is not None:
            logger.info('loaded pretrained model at {}'.format(configs.pretrained_path))
    else:
        if logger is not None:
            logger.info('Training from scratch - no pretrained weights loaded')

    # resume weights of model from a checkpoint
    if configs.resume_path is not None:
        assert os.path.isfile(configs.resume_path), "=> no checkpoint found at '{}'".format(configs.resume_path)
        model.load_state_dict(torch.load(configs.resume_path, map_location=configs.device))
        if logger is not None:
            logger.info('resume training model from checkpoint {}'.format(configs.resume_path))

    # Make sure to create optimizer after moving the model to cuda
    optimizer = create_optimizer(configs, model)
    lr_scheduler = create_lr_scheduler(optimizer, configs)
    configs.step_lr_in_epoch = True if configs.lr_type in ['multi_step'] else False

    # resume optimizer, lr_scheduler from a checkpoint
    if configs.resume_path is not None:
        utils_path = configs.resume_path.replace('Model_', 'Utils_')
        assert os.path.isfile(utils_path), "=> no checkpoint found at '{}'".format(utils_path)
        utils_state_dict = torch.load(utils_path, map_location=configs.device)
        optimizer.load_state_dict(utils_state_dict['optimizer'])
        lr_scheduler.load_state_dict(utils_state_dict['lr_scheduler'])
        configs.start_epoch = utils_state_dict['epoch'] + 1

    if configs.is_master_node:
        num_parameters = get_num_parameters(model)
        logger.info('number of trained parameters of the model: {}'.format(num_parameters))

    if logger is not None:
        logger.info(">>> Loading dataset & getting dataloader...")
    # Create dataloader
    train_dataloader, train_sampler = create_train_dataloader(configs)
    if logger is not None:
        logger.info('number of batches in training set: {}'.format(len(train_dataloader)))

    # Define your loss function (criterion)
    # criterion = torch.nn.MSELoss()

    if configs.evaluate:
        val_dataloader = create_val_dataloader(configs)
        precision, recall, AP, f1, ap_class = evaluate_mAP(val_dataloader, model, configs, None)
        # precision, recall, AP, f1, ap_class, val_loss = evaluate_mAP(val_dataloader, model, configs, logger, criterion)
        print('Evaluate - precision: {}, recall: {}, AP: {}, f1: {}, ap_class: {}'.format(precision, recall, AP, f1,
                                                                                          ap_class))

        print('mAP {}'.format(AP.mean()))
        return

    for epoch in range(configs.start_epoch, configs.num_epochs + 1):
        if logger is not None:
            logger.info('{}'.format('*-' * 40))
            logger.info('{} {}/{} {}'.format('=' * 35, epoch, configs.num_epochs, '=' * 35))
            logger.info('{}'.format('*-' * 40))
            logger.info('>>> Epoch: [{}/{}]'.format(epoch, configs.num_epochs))

        if configs.distributed:
            train_sampler.set_epoch(epoch)
        # train for one epoch
        train_one_epoch(train_dataloader, model, optimizer, lr_scheduler, epoch, configs, logger, tb_writer, scaler)
        if not configs.no_val:
            val_dataloader = create_val_dataloader(configs)
            print('number of batches in val_dataloader: {}'.format(len(val_dataloader)))
            precision, recall, AP, f1, ap_class = evaluate_mAP(val_dataloader, model, configs, logger)
            # precision, recall, AP, f1, ap_class, val_loss = evaluate_mAP(val_dataloader, model, configs, logger,
            #                                                              criterion)

            val_metrics_dict = {
                'precision': precision.mean(),
                'recall': recall.mean(),
                'AP': AP.mean(),
                'f1': f1.mean(),
                'ap_class': ap_class.mean()
            }

            # if tb_writer is not None:
            #     val_metrics_dict = {
            #         'precision': precision.mean(),
            #         'recall': recall.mean(),
            #         'AP': AP.mean(),
            #         'f1': f1.mean(),
            #         'ap_class': ap_class.mean(),
            #         'val_loss': val_loss  # Include the validation loss
            #     }
            if tb_writer is not None:
                tb_writer.add_scalars('Validation', val_metrics_dict, epoch)

            # Log the validation metrics to TensorBoard
            # if tb_writer is not None:
            #     for key, value in val_metrics_dict.items():
            #         tb_writer.add_scalar(f'Validation/{key}', value, epoch)

        # Save checkpoint
        # if configs.is_master_node and ((epoch % configs.checkpoint_freq) == 0):
        #     model_state_dict, utils_state_dict = get_saved_state(model, optimizer, lr_scheduler, epoch, configs)
        #     save_checkpoint(configs.checkpoints_dir, configs.saved_fn, model_state_dict, utils_state_dict, epoch)

        # Save checkpoint after every epoch
        if configs.is_master_node:
            model_state_dict, utils_state_dict = get_saved_state(model, optimizer, lr_scheduler, epoch, configs)
            save_checkpoint(configs.checkpoints_dir, configs.saved_fn, model_state_dict, utils_state_dict, epoch)

        if not configs.step_lr_in_epoch:
            lr_scheduler.step()
            if tb_writer is not None:
                tb_writer.add_scalar('LR', lr_scheduler.get_lr()[0], epoch)

    if tb_writer is not None:
        tb_writer.close()
    if configs.distributed:
        cleanup()


def cleanup():
    dist.destroy_process_group()


def train_one_epoch(train_dataloader, model, optimizer, lr_scheduler, epoch, configs, logger, tb_writer, scaler):
    """
    Train the model for one epoch
    """
    model.train()  # Set model to training mode
    
    # Initialize metrics
    total_loss = 0
    num_batches = len(train_dataloader)
    
    # Progress bar
    progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader),
                       desc=f'Training Epoch {epoch}', ncols=100)
    
    for batch_idx, batch_data in progress_bar:
        try:
            # Ensure batch_data is properly unpacked
            if not isinstance(batch_data, (tuple, list)) or len(batch_data) != 3:
                print(f"\nError: Unexpected batch_data format.")
                print(f"batch_data type: {type(batch_data)}")
                print(f"batch_data length: {len(batch_data) if hasattr(batch_data, '__len__') else 'N/A'}")
                print(f"batch_data contents: {batch_data}")
                continue
                
            paths, imgs, targets = batch_data
            
            # Debug information about tensors
            print(f"\nBatch {batch_idx + 1}/{num_batches}")
            print(f"Images shape: {imgs.shape}")
            print(f"Targets shape: {targets.shape if targets is not None else 'None'}")
            print(f"Number of images: {len(paths)}")
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            imgs = imgs.to(configs.device, non_blocking=True)
            targets = targets.to(configs.device, non_blocking=True)
            
            # Scale loss for gradient accumulation
            if scaler is not None:
                # MPS doesn't support autocast yet, just do regular forward pass
                loss, outputs = model(imgs, targets)
                loss = loss / configs.accumulation_steps
                scaler.scale(loss).backward()
            else:
                loss, outputs = model(imgs, targets)
                loss = loss / configs.accumulation_steps
                loss.backward()
            
            if (batch_idx + 1) % configs.accumulation_steps == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                
                if lr_scheduler is not None:
                    lr_scheduler.step()

            total_loss += loss.item()
            
            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({
                'batch_time': f'{time.time():.3f}s',
                'data_time': f'{time.time():.3f}s',
                'loss': f'{avg_loss:.4f}'
            })
            
        except Exception as e:
            print(f"\nError in batch {batch_idx}: {str(e)}")
            print(f"Stack trace:")
            import traceback
            traceback.print_exc()
            continue
    
    # Calculate average loss for the epoch
    avg_loss = total_loss / num_batches
    return avg_loss


if __name__ == '__main__':
    try:
        print("Starting training...")
        main()
    except KeyboardInterrupt:
        try:
            cleanup()
            sys.exit(0)
        except SystemExit:
            os._exit(0)
