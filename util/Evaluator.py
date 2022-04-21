import numpy as np


class Evaluator:
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((num_class,)*2)

    def acc(self):
        acc_base = np.sum(self.confusion_matrix, axis=0)
        # 避免分母为0
        acc_base[acc_base == 0] = 1
        acc_arr = np.diag(self.confusion_matrix) / acc_base
        return acc_arr, np.average(acc_arr)

    def oa(self):
        diag_val = np.diag(self.confusion_matrix)
        s = np.sum(self.confusion_matrix)
        return np.sum(diag_val) / s

    def iou(self):
        sum0 = np.sum(self.confusion_matrix, axis=0)
        sum1 = np.sum(self.confusion_matrix, axis=1)
        inter = np.diag(self.confusion_matrix)
        iou_arr = inter / (sum0 + sum1 - inter)
        return iou_arr, np.average(iou_arr)

    def iou_map(self):
        sum0 = np.sum(self.confusion_matrix, axis=0)
        sum1 = np.sum(self.confusion_matrix, axis=1)

        iou_map = np.zeros_like(self.confusion_matrix)
        for r in range(self.num_class):
            for c in range(self.num_class):
                iou_map[r][c] = self.confusion_matrix[r][c] / (sum0[r] + sum1[c] - self.confusion_matrix[r][c])
        return iou_map

    def recall(self):
        acc_base = np.sum(self.confusion_matrix, axis=1)
        # 避免分母为0
        acc_base[acc_base == 0] = 1
        recall_arr = np.diag(self.confusion_matrix) / acc_base
        return recall_arr, np.average(recall_arr)

    def f1_score(self):
        acc_arr, avg_acc = self.acc()
        recall_arr, avg_recall = self.recall()
        f1_base = acc_arr + recall_arr
        f1_base[f1_base == 0] = 1
        f1 = 2 * acc_arr * recall_arr / f1_base
        return f1, np.average(f1)

    def fw_iou(self):
        s = np.sum(self.confusion_matrix)
        freq_num = np.sum(self.confusion_matrix, axis=1)
        freq = freq_num/s
        iou, iou_avg = self.iou()
        fw = freq*iou
        return fw, np.sum(fw)

    def add_batch(self, batch_confusion_matrix):
        assert self.confusion_matrix.shape == batch_confusion_matrix.shape
        self.confusion_matrix += batch_confusion_matrix

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
