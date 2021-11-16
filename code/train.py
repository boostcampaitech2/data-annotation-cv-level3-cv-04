import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST
from loss import EASTLoss
import wandb


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '../input/data/ICDAR17_Korean'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--input_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--save_interval', type=int, default=5)

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args

# 추가 부분 -> json으로 거름
# def my_collate(batch):
#     batch = filter(lambda x: x[0] is not None, batch) # None이 있는 경우, Batch에서 빼버림
#     return torch.utils.data.dataloader.default_collate(list(batch))

def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval):
    
    # train data
    dataset = SceneTextDataset(data_dir, split='splited_train', image_size=image_size, crop_size=input_size)
    train_dataset = EASTDataset(dataset)
    
    # val data
    dataset = SceneTextDataset(data_dir, split='splited_val', image_size=image_size, crop_size=input_size)
    val_dataset = EASTDataset(dataset)
    
    num_batches = math.ceil(len(train_dataset) / batch_size)
    val_num_batches = math.ceil(len(val_dataset) / batch_size)
    
    # train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=my_collate)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)
    criterion = EASTLoss()
    
    wandb.watch(model)
    
    for epoch in range(max_epoch):
        # train
        model.train()
        epoch_loss, epoch_start = [], time.time()
        train_cls_loss, train_angle_loss, train_iou_loss, train_mean_loss = [], [], [], []
        with tqdm(total=num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description('[Train Epoch {}]'.format(epoch + 1))

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss.append(loss_val)

                pbar.update(1)
                val_dict = {
                    'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss']
                }
                
                
                pbar.set_postfix(val_dict)
                
                if extra_info['cls_loss'] != None: #  positive sample(pixel)이 아예 없는 경우가 발생 가능
                    train_cls_loss.append(extra_info['cls_loss'])
                    train_angle_loss.append(extra_info['angle_loss'])
                    train_iou_loss.append(extra_info['iou_loss'])

        print('Train Mean loss: {:.4f} | Elapsed time: {}'.format(sum(epoch_loss) / len(epoch_loss), timedelta(seconds=time.time() - epoch_start)))        
        train_mean_loss = sum(epoch_loss) / len(epoch_loss)
        
        # validation
        model.eval()
        with torch.no_grad():
            epoch_loss, epoch_start = [], time.time()
            val_cls_loss, val_angle_loss, val_iou_loss, val_mean_loss = [], [], [], []
            with tqdm(total=val_num_batches) as pbar:
                for img, gt_score_map, gt_geo_map, roi_mask in val_loader:
                    pbar.set_description('[Val Epoch {}]'.format(epoch + 1))

                    # 평가
                    image, score_map, geo_map, roi_mask = (img.to(device), gt_score_map.to(device), gt_geo_map.to(device), roi_mask.to(device))
                    pred_score_map, pred_geo_map = model(image)
                    loss, values_dict = criterion(score_map, pred_score_map, geo_map, pred_geo_map, roi_mask)
                    extra_info = dict(**values_dict, score_map=pred_score_map, geo_map=pred_geo_map)

                    loss_val = loss.item()
                    epoch_loss.append(loss_val)

                    pbar.update(1)
                    val_dict = {
                        'Val Cls loss': extra_info['cls_loss'], 'Val Angle loss': extra_info['angle_loss'],
                        'Val IoU loss': extra_info['iou_loss']
                    }
                    
                    pbar.set_postfix(val_dict)
                    
                    if extra_info['cls_loss'] != None: #  positive sample(pixel)이 아예 없는 경우가 발생 가능
                        val_cls_loss.append(extra_info['cls_loss'])
                        val_angle_loss.append(extra_info['angle_loss'])
                        val_iou_loss.append(extra_info['iou_loss'])

            print('Val Mean loss: {:.4f} | Elapsed time: {}'.format(sum(epoch_loss) / len(epoch_loss), timedelta(seconds=time.time() - epoch_start)))
            val_mean_loss = sum(epoch_loss) / len(epoch_loss)
            
        wandb.log({
                    "train/loss" : train_mean_loss,
                    "train/Cls_loss": (sum(train_cls_loss) / len(train_cls_loss)),
                    "train/Angle_loss" : (sum(train_angle_loss) / len(train_angle_loss)),
                    "train/IoU_loss" : (sum(train_iou_loss) / len(train_iou_loss)),
                    "val/loss" : val_mean_loss,
                    "val/Cls_loss": (sum(val_cls_loss) / len(val_cls_loss)),
                    "val/Angle_loss" : (sum(val_angle_loss) / len(val_angle_loss)),
                    "val/IoU_loss" : (sum(val_iou_loss) / len(val_iou_loss)),
                    })
        
        scheduler.step()

        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            ckpt_fpath = osp.join(model_dir, 'latest.pth')
            torch.save(model.state_dict(), ckpt_fpath)

def main(args):
    
    wandb.init(
        project='optical-character-recognition',
        entity='cv4',
        name='train_val_polygon'
    )
    
    wandb.config.update(args)
    
    do_training(**args.__dict__)


if __name__ == '__main__':
    args = parse_args()
    main(args)
