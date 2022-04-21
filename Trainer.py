import os
import cv2 as cv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import get_model
from dataset import get_dataloader
from optim import get_optim

from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from BluePrint import BluePrint
from util import Now, Logger, Evaluator
from util import CheckpointManager
import json


class Trainer:
    def _get_model(self, config):
        model_name = config['model']
        model = get_model(model_name)
        if self.arch == 'gpu':
            model = nn.DataParallel(model).cuda()
        sd = self.model_checkpoint_manager.load_best()
        # sd = self.model_checkpoint_manager.load_by_name('model_20210819192613.pkl')
        if sd is not None:
            model.load_state_dict(sd)
            self.logger.info('load checkpoint - {}!'.format(self.model_checkpoint_manager.get_last_loaded_checkpoint()))
        return model

    def __init__(self, config):
        self.blueprint = BluePrint()
        self.config = config
        self.num_classes = config['model']['ch_out']
        model_name = config['model']['name']
        checkpoint_name = model_name
        if 'backbone' in config['model']:
            back = config['model']['backbone']
            checkpoint_name = f'{model_name}_{back}'
        self.arch = config['run']['arch']
        # logger
        self.logger = Logger.get_logger(log_file=self.blueprint.LOG_MAIN.touch_(checkpoint_name))
        self.model_checkpoint_manager = CheckpointManager(log_root=self.blueprint.MODEL_CHECKPOINT.touch(),
                                                          model_type=checkpoint_name)
        # model
        self.model = self._get_model(config)
        # optim
        self.optimizer = get_optim(self.model, config['optim'])
        # loss fn
        self.loss_fn_ce = nn.CrossEntropyLoss(reduction='mean')
        # dataset
        self.train_loader = get_dataloader(config['dataset'], subset='train')
        self.val_loader = get_dataloader(config['dataset'], subset='val')
        self.test_loader = get_dataloader(config['dataset'], subset='test')

    def train(self):
        n_class = np.arange(0, self.num_classes)
        evaluator = Evaluator(self.num_classes)
        # begin to train
        epochs = self.config['run']['train']['num_epoch']
        for epoch in range(epochs):
            # train
            self.model.train()
            with tqdm(total=len(self.train_loader), desc=f'epoch {epoch + 1}/{epochs}', unit='itr') as pbar:
                loss_val = 0
                for batch in self.train_loader:
                    # print(batch['name'])
                    img, gt = batch['img'], batch['gt']
                    if self.arch == 'gpu':
                        img, gt = batch['img'].cuda(), batch['gt'].cuda()
                    gt = gt.to(dtype=torch.long)
                    out = self.model(img)

                    loss = self.loss_fn_ce(out, gt)

                    self.optimizer.zero_grad()
                    loss_val += loss.item()
                    loss.backward()
                    self.optimizer.step()
                    pbar.update()

                    # relu
                    gt = gt.detach().cpu().numpy()
                    out = torch.max(out, dim=1)[1].clone().detach().cpu().numpy()
                    cm = confusion_matrix(gt.flatten(), out.flatten(), labels=n_class)
                    evaluator.add_batch(cm)

            avg_loss = loss_val / len(self.train_loader)
            oa = evaluator.oa()
            _, iou = evaluator.iou()
            _, fw_iou = evaluator.fw_iou()

            ret = {
                'loss': avg_loss,
                'oa': oa,
                'iou': iou,
                'fw_iou': fw_iou
            }
            self.logger.info('Train epoch {} : {}'.format(epoch + 1, json.dumps(ret)))
            # validate
            self.validate(epoch)

    def validate(self, epoch=0):
        n_class = np.arange(0, self.num_classes)
        evaluator = Evaluator(self.num_classes)
        out_pred_dir = self.blueprint.VAL_SEG_OUTPUT.touch()

        self.model.eval()
        with tqdm(total=len(self.val_loader), desc=f'val {epoch + 1}', unit='itr') as pbar:
            loss_val = 0
            for batch in self.val_loader:
                img, gt = batch['img'], batch['gt']
                if self.arch == 'gpu':
                    img, gt = batch['img'].cuda(), batch['gt'].cuda()
                gt_ = gt.to(dtype=torch.long)
                out = self.model(img)

                loss = self.loss_fn_ce(out, gt_)
                loss_val += loss.item()
                # relu
                gt = gt.detach().cpu().numpy()
                out = torch.max(out, dim=1)[1].clone().detach().cpu().numpy()
                cm = confusion_matrix(gt.flatten(), out.flatten(), labels=n_class)
                evaluator.add_batch(cm)

                # save pic
                if True:
                    for idx in range(len(gt)):
                        pic_name = batch['name'][idx]
                        out_idx = out[idx]
                        out_idx[out_idx>0] = 255
                        out_pred_path = os.path.join(out_pred_dir, pic_name)
                        cv.imwrite(out_pred_path, out_idx)

                pbar.update()

        avg_loss = loss_val / len(self.val_loader)
        oa = evaluator.oa()
        _, iou = evaluator.iou()
        _, fw_iou = evaluator.fw_iou()
        ret = {
            'loss': avg_loss,
            'oa': oa,
            'iou': iou,
            'fw_iou': fw_iou
        }
        self.logger.info('VAL epoch {} : {}'.format(epoch + 1, json.dumps(ret)))

        # save model
        self.model_checkpoint_manager.save(self.model, avg_loss, oa)

    def test(self):
        ch_out = self.config['model']['ch_out']
        model_name = self.config['model']['name']

        evaluator = Evaluator(ch_out)
        self.logger.info('{}'.format(model_name))
        out_pred_dir = self.blueprint.TEST_SEG_OUTPUT.touch()
        self.model.eval()

        n_class = np.arange(0, ch_out)
        with tqdm(total=len(self.val_loader), desc=f'test', unit='itr') as pbar:
            for batch in self.val_loader:
                img, gt = batch['img'], batch['gt']
                if self.arch == 'gpu':
                    img, gt = batch['img'].cuda(), batch['gt']
                out = self.model(img)
                # relu
                out = torch.max(out, dim=1)[1].clone().detach().cpu().numpy()
                gt = gt.detach().cpu().numpy()
                cm = confusion_matrix(gt.flatten(), out.flatten(), labels=n_class)
                evaluator.add_batch(cm)

                # save pic
                if True:
                    for idx in range(len(gt)):
                        pic_name = batch['name'][idx]
                        out_pred_path = os.path.join(out_pred_dir, pic_name)
                        cv.imwrite(out_pred_path, out[idx])
                        # save weight
                        # out_weight_path = os.path.join(out_weight_dir, pic_name)
                        # parser_w = weight[idx][0].detach().cpu().numpy() * 255
                        # parser_w = parser_w.astype(np.uint8)
                        # cv.imwrite(out_weight_path, parser_w)

                pbar.update()
                # break

        # count acc
        acc_arr, m_acc = evaluator.acc()
        acc_str = ' , '.join(str(a) for a in acc_arr)
        self.logger.info('acc: {} OA: {}'.format(acc_str, m_acc))
        # count IoU
        iou_arr, mean_iou = evaluator.iou()
        iou_str = ' , '.join(str(a) for a in iou_arr)
        self.logger.info('iou: {} mIoU: {}'.format(iou_str, mean_iou))
        # count fw_iou
        fw_iou, mean_fw_iou = evaluator.fw_iou()
        fw_iou_str = ' , '.join(str(a) for a in fw_iou)
        self.logger.info('fw_iou: {} mF1: {}'.format(fw_iou_str, mean_fw_iou))

    def predict(self):
        model_name = self.config['model']['name']

        self.logger.info('{}'.format(model_name))
        out_pred_dir = self.config['dataset']['root_out']
        self.model.eval()

        with tqdm(total=len(self.test_loader), desc=f'test', unit='itr') as pbar:
            for batch in self.test_loader:
                img = batch['img'].cuda() if self.arch == 'gpu' else batch['img']
                out = self.model(img)

                # save pic
                if True:
                    for idx in range(len(out)):
                        pic_name = batch['name'][idx]
                        out_idx = out[idx]

                        # 根据图片大小存储
                        _, p_w, p_h = out_idx.size()
                        raw_size = batch['size'][idx].split(',')
                        raw_w, raw_h = int(raw_size[0]), int(raw_size[1])
                        if p_w != raw_w and p_h != raw_h:
                            out_idx = torch.unsqueeze(out_idx, dim=0)
                            out_idx = F.interpolate(out_idx, size=[raw_h, raw_w], mode='bilinear')[0]

                        out_pic = torch.max(out_idx, dim=0)[1].clone().detach().cpu().numpy()
                        out_pic[out_pic > 0] = 255
                        out_pred_path = os.path.join(out_pred_dir, pic_name)
                        cv.imwrite(out_pred_path, out_pic)

                pbar.update()
