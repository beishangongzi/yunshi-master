import os
import cv2 as cv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
import torch.nn as nn

from models import get_model
from dataset import get_dataloader
from optim import get_optim

from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns

from BluePrint import BluePrint
from util import Now, Logger, Evaluator
from util import CheckpointManager


class Eval:
    def _get_model(self, config):
        model_name = config['model']
        model = get_model(model_name)
        if self.arch == 'gpu':
            model = nn.DataParallel(model).cuda()
        sd = self.model_checkpoint_manager.load_best()
        # sd = self.model_checkpoint_manager.load_by_name('model_20210819192613.pkl')
        if sd is not None:
            model.load_state_dict(sd)
            self.logger.info('load checkpoint - {}!'.format(self.model_checkpoint_manager.model_best_path()))
        return model

    def __init__(self, config):
        self.blueprint = BluePrint()
        self.config = config
        model_name = config['model']['name']
        self.arch = config['run']['arch']
        # logger
        self.logger = Logger.get_logger(log_file=self.blueprint.LOG_MAIN.touch_(model_name))
        self.model_checkpoint_manager = CheckpointManager(log_root=self.blueprint.MODEL_CHECKPOINT.touch(),
                                                          model_type=model_name)
        # model
        self.model = self._get_model(config)
        # optim
        self.optimizer = get_optim(self.model, config['optim'])
        # loss fn
        self.loss_fn_ce = nn.CrossEntropyLoss(reduction='mean')
        # dataset
        self.train_loader = get_dataloader(config['dataset'], subset='train')

    def train(self):
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

            avg_loss = loss_val / len(self.train_loader)
            self.logger.info('epoch {} avg loss is {}'.format(epoch + 1, avg_loss))
            # validate
            self.validate(epoch)

            if avg_loss < 0.01:
                self.logger.info('early stop!')
                break

    def validate(self, epoch=0):
        ch_out = self.config['model']['ch_out']
        n_class = np.arange(0, ch_out)
        evaluator = Evaluator(ch_out)
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
                        out_pred_path = os.path.join(out_pred_dir, pic_name)
                        cv.imwrite(out_pred_path, out[idx])

                pbar.update()

        avg_loss = loss_val / len(self.val_loader)
        self.logger.info('val avg_loss is {}'.format(avg_loss))

        # save model
        self.model_checkpoint_manager.save(self.model, avg_loss)

    def test(self):
        ch_out = self.config['model']['ch_out']
        model_name = self.config['model']['name']

        evaluator = Evaluator(ch_out)
        self.logger.info('{}'.format(model_name))
        out_pred_dir = self.blueprint.TEST_SEG_OUTPUT.touch()
        self.model.eval()

        n_class = np.arange(0, ch_out)
        with tqdm(total=len(self.train_loader), desc=f'test', unit='itr') as pbar:
            for batch in self.train_loader:
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
