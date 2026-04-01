# --- import bootstrap: robust no matter CWD / runner ---
import os, sys, pathlib
HERE = pathlib.Path(__file__).resolve()
PKG_ROOT = HERE.parent                      # /.../mambacode/muti_mamba
REPO_ROOT = PKG_ROOT.parent                 # /.../mambacode
for p in (str(PKG_ROOT), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
# -------------------------------------------------------

import numpy as np

import torch
import torch.nn as nn
from monai.inferers import SlidingWindowInferer
# from muti_mamba.light_training.dataloading.dataset import get_train_val_test_loader_from_train_muti_modal
# from muti_mamba.light_training.evaluation.metric import dice
# from muti_mamba.light_training.trainer import Trainer
# from muti_mamba.light_training.utils.files_helper import save_new_model_and_delete_last
from monai.utils import set_determinism

from monai.losses.dice import DiceLoss
set_determinism(123)
import os

from muti_mamba.light_training.dataloading.dataset import get_train_val_test_loader_from_train_muti_modal
from muti_mamba.light_training.evaluation.metric import dice
from muti_mamba.light_training.trainer import Trainer
from muti_mamba.light_training.utils.files_helper import save_new_model_and_delete_last


# data_dir = "./data/fullres/RHUH_NPZ"
# data_dir = "./data/fullres/UCSF_4"
#data_dir = "./data/fullres/SSA_npz"
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
data_dir = "./data/fullres/BraTS2020_NPZ"
# data_dir = "./data/fullres/SSA"
# data_dir = "./data/fullres/processed_4state/processed"


# logdir = f"./logs/muti_modal"
logdir = f"./logs/brats2020"
# logdir = f"./logs/ssa"
# logdir = f"./logs/upenn"

model_save_path = os.path.join(logdir, "model")
# augmentation = "nomirror"
augmentation = True

env = "pytorch"
max_epoch = 400
batch_size = 2
val_every = 2
num_gpus = 1
device = "cuda:0"
roi_size = [80, 160, 160]


def func(m, epochs):
    return np.exp(-10 * (1 - m / epochs) ** 2)
import numpy as np
from medpy.metric import binary


def calculate_hd95(output, label, voxel_spacing=[1.0, 1.0, 1.0]):
    batch_size, num_classes, D, W, H = output.shape  # 获取 output 和 label 的维度
    hd95s = []

    # 遍历每个类别 (类别数为 3: TC, WT, ET)
    for i in range(num_classes):
        # 获取第 i 类别的预测和真实标签
        pred_c = output[:, i]  # (batch_size, D, W, H) -- 预测类别标签
        target_c = label[:, i]  # (batch_size, D, W, H) -- 真实类别标签

        # 确保 pred_c 和 target_c 仅包含二进制值（0 或 1）
        pred_c = (pred_c > 0.5).astype(np.uint8)  # 将概率值转换为 0 或 1，确保类型为 uint8
        target_c = (target_c > 0.5).astype(np.uint8)  # 将标签值转换为 0 或 1，确保类型为 uint8

        # 计算每个批次中的 HD95
        batch_hd95 = []
        for b in range(batch_size):
            # 提取当前批次的第 i 类别的预测和标签
            pred_c_b = pred_c[b]  # 预测类别标签，已经是二进制的
            target_c_b = target_c[b]  # 真实类别标签，已经是二进制的

            # 确保数据没有 NaN 或无效值
            pred_c_b = np.nan_to_num(pred_c_b)  # 替换 NaN 为 0
            target_c_b = np.nan_to_num(target_c_b)  # 替换 NaN 为 0

            # 检查是否 target_c_b 中有 1（如果没有 1，就跳过计算 HD95）
            # 检查是否 pred_c_b 或 target_c_b 全为 0 或全为 1
            if np.all(pred_c_b == 0) or np.all(target_c_b == 0):  # 全为0
                print(f"Warning: Predicted or target for class {i} in batch {b} is all 0. Skipping HD95 calculation.")
                batch_hd95.append(np.nan)  # 选择一个占位符值（如 NaN），表示无法计算 HD95
                return 0
            elif np.all(pred_c_b == 1) or np.all(target_c_b == 1):  # 全为1
                print(f"Warning: Predicted or target for class {i} in batch {b} is all 1. Skipping HD95 calculation.")
                batch_hd95.append(np.nan)  # 选择一个占位符值（如 NaN），表示无法计算 HD95
                return 0
            else:
                # 计算 HD95 (假设 binary.hd95 计算的是二进制分割的 HD95)
                hd95 = binary.hd95(target_c_b, pred_c_b, voxelspacing=voxel_spacing)
                batch_hd95.append(hd95)

        # 计算当前类别的平均 HD95（批次内的平均值），忽略 NaN
        valid_hd95s = [hd for hd in batch_hd95 if not np.isnan(hd)]  # 忽略 NaN
        if valid_hd95s:
            hd95s.append(np.mean(valid_hd95s))
        else:
            hd95s.append(np.nan)  # 如果没有有效的 HD95 值，返回 NaN

    return hd95s

class BraTSTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/",
                 master_ip='localhost', master_port=17750, training_script="train.py", pretrained_weights=None):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port,
                         training_script)
        self.hd95_global = []
        self.window_infer = SlidingWindowInferer(roi_size=roi_size,
                                                 sw_batch_size=1,
                                                 overlap=0.5)
        self.augmentation = augmentation
        
        from mprm_mamba.model.mprm_mamba import MprmMamba
        

        self.model = MprmMamba(in_chans=1,
                              out_chans=4,
                              depths=[2, 2, 2, 2],
                              feat_size=[48, 96, 192, 384])
        # 加载预训练权重
        # if pretrained_weights:
        #     self.model.load_state_dict(torch.load(pretrained_weights), strict=False)
        # state_dict = torch.load(pretrained_weights)
        # model_state_dict = self.model.state_dict()
        #
        # # 保留匹配的参数
        # new_state_dict = {k: v for k, v in state_dict.items() if
        #                   k in model_state_dict and model_state_dict[k].shape == v.shape}
        # model_state_dict.update(new_state_dict)
        # self.model.load_state_dict(model_state_dict)
        self.patch_size = roi_size
        self.best_mean_dice = 0.0
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.train_process = 18
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3, weight_decay=3e-5,
                                         momentum=0.99, nesterov=True)

        self.scheduler_type = "poly"
        self.cross = nn.CrossEntropyLoss()

    def training_step(self, batch):
        image, label, patient_ids = self.get_input(batch)
        feat1 = image[:, 0:1, :, :, :]  # 第一个特征
        feat2 = image[:, 1:2, :, :, :]  # 第二个特征
        feat3 = image[:, 2:3, :, :, :]  # 第三个特征
        feat4 = image[:, 3:4, :, :, :]  # 第四个特征
        pred = self.model(feat1, feat2, feat3, feat4)#2*4*80*160*160, 2*80*160*160

        loss = self.cross(pred, label)

        self.log("training_loss", loss, step=self.global_step)

        return loss

    def convert_labels(self, labels):
        ## TC, WT and ET
        result = [(labels == 1) | (labels == 3), (labels == 1) | (labels == 3) | (labels == 2), labels == 3]

        return torch.cat(result, dim=1).float()

    def get_input(self, batch):
        image = batch["data"]
        label = batch["seg"]
        properties = batch['properties']  # 每个样本的属性列表
        patient_ids = [prop['patient_id'] for prop in properties]  # 从每个属性字典中提取 patient_id
        label = label[:, 0].long()
        label[label == 4] = 3

        return image, label, patient_ids

    def cal_metric(self, gt, pred, voxel_spacing=[1.0, 1.0, 1.0]):
        if pred.sum() > 0 and gt.sum() > 0:
            d = dice(pred, gt)
            return np.array([d, 50])

        elif gt.sum() == 0 and pred.sum() == 0:
            return np.array([1.0, 50])

        else:
            return np.array([0.0, 50])

    def validation_step(self, batch):
        image, label, patient_ids = self.get_input(batch)
        feat1 = image[:, 0:1, :, :, :]  # 第一个特征
        feat2 = image[:, 1:2, :, :, :]  # 第二个特征
        feat3 = image[:, 2:3, :, :, :]  # 第三个特征
        feat4 = image[:, 3:4, :, :, :]  # 第四个特征
        output = self.model(feat1, feat2, feat3, feat4)

        output = output.argmax(dim=1)

        output = output[:, None]
        output = self.convert_labels(output)#2*3*80*160*160

        label = label[:, None]
        label = self.convert_labels(label)

        output = output.cpu().numpy()
        target = label.cpu().numpy()
        # print(np.unique(output))  # 检查输出中是否有非二进制值
        # print(np.unique(target))  # 检查标签中是否有非二进制值

        dices = []
        c = 3
        for i in range(0, c):
            pred_c = output[:, i]
            target_c = target[:, i]

            cal_dice, _ = self.cal_metric(target_c, pred_c)
            dices.append(cal_dice)
        # print("dices",dices)
        hd95s = calculate_hd95(output, target)
        if hd95s != 0:
            self.hd95_global.append(hd95s)

        # print(hd95s)
        return dices

    def validation_end(self, val_outputs):
        dices = val_outputs
        # 拼接hd95_global
        hd95s_all = np.stack(self.hd95_global, axis=0)  # 按batch维度拼起来
        hd95s_mean = np.mean(hd95s_all, axis=0)  # 每个类别求平均

        tc, wt, et = dices[0].mean(), dices[1].mean(), dices[2].mean()
        tc_hd, wt_hd, et_hd = hd95s_mean[0], hd95s_mean[1], hd95s_mean[2]

        print(f"dices is {tc, wt, et}")
        print(f"hd95 is {tc_hd, wt_hd, et_hd}")

        mean_dice = (tc + wt + et) / 3
        mean_hd = (tc_hd + wt_hd + et_hd) / 3
        self.log("tc", tc, step=self.epoch)
        self.log("wt", wt, step=self.epoch)
        self.log("et", et, step=self.epoch)

        self.log("mean_dice", mean_dice, step=self.epoch)

        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice
            # save_new_model_and_delete_last(self.model,
            #                                os.path.join(model_save_path,
            #                                             f"RU_FT_best_model_{mean_dice:.4f}.pt"),
            #                                delete_symbol="RU_FT_best_model_")
            save_new_model_and_delete_last(self.model,
                                           os.path.join(model_save_path,
                                                        f"RHUH_FT_best_model_{mean_dice:.4f}.pt"),
                                           delete_symbol="RHUH_FT_best_model_")

        # save_new_model_and_delete_last(self.model,
        #                                os.path.join(model_save_path,
        #                                             f"RU_FT_final_model_{mean_dice:.4f}.pt"),
        #                                delete_symbol="RU_FT_final_model")
        save_new_model_and_delete_last(self.model,
                                       os.path.join(model_save_path,
                                                    f"RHUH_FT_final_model_{mean_dice:.4f}.pt"),
                                       delete_symbol="RHUH_FT_final_model")
        if (self.epoch + 1) % 100 == 0:
            torch.save(self.model.state_dict(),
                       os.path.join(model_save_path, f"tmp_model_ep{self.epoch}_{mean_dice:.4f}.pt"))
        print(f"mean_hd is {mean_hd}")

        print(f"mean_dice is {mean_dice}")
        # 用完之后清空
        self.hd95_global = []

if __name__ == "__main__":
    trainer = BraTSTrainer(env_type=env,
                           max_epochs=max_epoch,
                           batch_size=batch_size,
                           device=device,
                           logdir=logdir,
                           val_every=val_every,
                           num_gpus=num_gpus,
                           master_port=17759,
                           training_script=__file__,
                           pretrained_weights=pretrained_weights_path)

    train_ds, val_ds, test_ds = get_train_val_test_loader_from_train_muti_modal(data_dir)

    trainer.train(train_dataset=train_ds, val_dataset=val_ds)
    
