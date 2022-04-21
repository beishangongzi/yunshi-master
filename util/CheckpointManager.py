import torch
import numpy as np
import os
import sys
from util.Logger import Logger
from util.Now import Now
from util.FileExtFilter import FileExtFilter


class CheckpointManager:
    def _load_ref_info(self):
        loss = sys.maxsize
        acc = 0
        if os.path.exists(self.loss_fp):
            arr = np.load(self.loss_fp)
            if arr[0] < loss:
                loss = arr[0]
            if len(arr) > 1 and arr[1] > acc:
                acc = arr[1]
        return loss, acc

    def check_dir(self, dir):
        if not os.path.exists(dir):
            os.mkdir(dir)

    def __init__(self, log_root='log', model_type='unet_321', loss_fn='loss.npy'):
        # basic
        self.root = os.path.join(log_root, model_type)
        self.loss_fp = os.path.join(self.root, loss_fn)
        self.model_loss_min_fp = os.path.join(self.root, 'loss_min.pth')
        self.model_acc_max_fp = os.path.join(self.root, 'acc_max.pth')

        self.check_dir(self.root)

        log_fn = os.path.join(self.root, 'checkpoint.log')
        self.logger = Logger.get_file_logger('ck', log_file=log_fn)

        # load last loss
        self.loss, self.acc = self._load_ref_info()

        self.last_loaded_checkpoint = ''

    def save(self, model, loss, acc):
        # save model
        model_fn = 'model_{}.pkl'.format(Now.current_dt())
        model_fp = os.path.join(self.root, model_fn)
        torch.save(model.state_dict(), model_fp)
        self.logger.info('model : {} - loss is : {} acc is : {}'.format(model_fn, loss, acc))

        # save min loss model
        self._save_best(model, loss, acc)

        # save max acc model
        self.clean_pkl(threshold=50)

    def load_best(self, loss_min=True, acc_max=False):
        assert loss_min ^ acc_max

        if loss_min and os.path.exists(self.model_loss_min_fp):
            print('load checkpoint: ' + self.model_loss_min_fp)
            sd = torch.load(self.model_loss_min_fp)
            self.last_loaded_checkpoint = self.model_loss_min_fp
            return sd

        if acc_max and os.path.exists(self.model_acc_max_fp):
            print('load checkpoint: ' + self.model_acc_max_fp)
            sd = torch.load(self.model_acc_max_fp)
            self.last_loaded_checkpoint = self.model_acc_max_fp
            return sd

        return None

    def load_by_name(self, name):
        fp = os.path.join(self.root, name)
        if os.path.exists(fp):
            sd = torch.load(fp)
            self.last_loaded_checkpoint = fp
            return sd
        return None

    def get_last_loaded_checkpoint(self):
        return self.last_loaded_checkpoint

    def remove_pth(self, model_name):
        if os.path.exists(model_name):
            os.remove(model_name)

    def _save_best(self, model, loss, acc):
        arr = [0, 0]
        required_to_save = False
        # 处理loss
        if loss < self.loss:
            self.remove_pth(self.model_loss_min_fp)
            # start to create
            torch.save(model.state_dict(), self.model_loss_min_fp)
            arr[0] = loss
            self.loss = loss
            required_to_save = True

        # 处理acc
        if acc > self.acc:
            self.remove_pth(self.model_acc_max_fp)
            # start to create
            torch.save(model.state_dict(), self.model_acc_max_fp)
            arr[1] = acc
            self.acc = acc
            required_to_save = True

        if required_to_save:
            np.save(self.loss_fp, arr)

    def clean_pkl(self, threshold=5):
        files = os.listdir(self.root)
        extFilter = FileExtFilter(ext=['pkl'])
        files = list(filter(extFilter.filter, files))
        if (len(files) > threshold):
            for file in files:
                fp = os.path.join(self.root, file)
                os.remove(fp)


if __name__ == '__main__':
    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.conv1 = torch.nn.Conv2d(1, 3, 2)

    model = Model()
    ck = CheckpointManager()
    # ck.save(model, 0.8)
    # ck.save(model, 0.9, 4)
    # sd = ck.load_best()
    # print(1)