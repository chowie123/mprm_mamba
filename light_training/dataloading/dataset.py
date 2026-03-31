
# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from sklearn.model_selection import KFold  ## K折交叉验证
import pickle
import os
import json
import math
import numpy as np
import torch
from monai import transforms
import SimpleITK as sitk
from tqdm import tqdm 
from torch.utils.data import Dataset 
import glob 
from muti_mamba.light_training.dataloading.utils import unpack_dataset
import random 

class MedicalDataset(Dataset):
    def __init__(self, datalist, test=False) -> None:
        super().__init__()
        
        self.datalist = datalist
        self.test = test 

        self.data_cached = []
        for p in tqdm(self.datalist, total=len(self.datalist)):
            info = self.load_pkl(p)

            self.data_cached.append(info)

        ## unpacking
        print(f"unpacking data ....")
        # for 
        folder = []
        for p in self.datalist:
            f = os.path.dirname(p)
            if f not in folder:
                folder.append(f)
        for f in folder:
            unpack_dataset(f, 
                        unpack_segmentation=True,
                        overwrite_existing=False,
                        num_processes=8)


        print(f"data length is {len(self.datalist)}")
        
    def load_pkl(self, data_path):
        pass 
        properties_path = f"{data_path[:-4]}.pkl"
        df = open(properties_path, "rb")
        info = pickle.load(df)

        return info 
    
    def post(self, batch_data):
        return batch_data
    
    def read_data(self, data_path):
        
        image_path = data_path.replace(".npz", ".npy")
        seg_path = data_path.replace(".npz", "_seg.npy")
        image_data = np.load(image_path, "r+")
      
        seg_data = None 
        if not self.test:
            seg_data = np.load(seg_path, "r+")
        return image_data, seg_data

    def __getitem__(self, i):
        
        image, seg = self.read_data(self.datalist[i])

        properties = self.data_cached[i]

        if seg is None:
            return {
                "data": image,
                "properties": properties
            }
        else :
            return {
                "data": image,
                "seg": seg,
                "properties": properties
            }

    def __len__(self):
        return len(self.datalist)

def get_train_test_loader_from_test_list(data_dir, test_list):
    all_paths = glob.glob(f"{data_dir}/*.npz")

    test_datalist = []
    train_datalist = []

    test_list_1 = []
    for t in test_list:
        test_list_1.append(t.replace(".nii.gz", ""))

    test_list = test_list_1
    for p in all_paths:
        p2 = p.split("/")[-1].split(".")[0]
        if p2 in test_list:
            test_datalist.append(p)
        else :
            train_datalist.append(p)

    print(f"training data is {len(train_datalist)}")
    print(f"test data is {len(test_datalist)}", test_datalist)

    train_ds = MedicalDataset(train_datalist)
    test_ds = MedicalDataset(test_datalist)

    loader = [train_ds, test_ds]

    return loader

def get_kfold_data(data_paths, n_splits, shuffle=False):
    X = np.arange(len(data_paths))
    kfold = KFold(n_splits=n_splits, shuffle=shuffle)  ## kfold为KFolf类的一个对象
    return_res = []
    for a, b in kfold.split(X):
        fold_train = []
        fold_val = []
        for i in a:
            fold_train.append(data_paths[i])
        for j in b:
            fold_val.append(data_paths[j])
        return_res.append({"train_data": fold_train, "val_data": fold_val})

    return return_res

def get_kfold_loader(data_dir, fold=0, test_dir=None):

    all_paths = glob.glob(f"{data_dir}/*.npz")
    fold_data = get_kfold_data(all_paths, 5)[fold]

    train_datalist = fold_data["train_data"]
    val_datalist = fold_data["val_data"]  

    print(f"training data is {len(train_datalist)}")
    print(f"validation data is {len(val_datalist)}")
    train_ds = MedicalDataset(train_datalist)
    
    val_ds = MedicalDataset(val_datalist)

    if test_dir is not None:
        test_paths = glob.glob(f"{test_dir}/*.npz")
        test_ds = MedicalDataset(test_paths, test=True)
    else:
        test_ds = None 

    loader = [train_ds, val_ds, test_ds]

    return loader

def get_all_training_loader(data_dir, fold=0, test_dir=None):
    ## train all labeled data 
    ## fold denote the validation data in training data
    all_paths = glob.glob(f"{data_dir}/*.npz")
    fold_data = get_kfold_data(all_paths, 5)[fold]

    train_datalist = all_paths
    val_datalist = fold_data["val_data"]  

    print(f"training data is {len(train_datalist)}")
    print(f"validation data is {len(val_datalist)}")
    train_ds = MedicalDataset(train_datalist)
    
    val_ds = MedicalDataset(val_datalist)

    if test_dir is not None:
        test_paths = glob.glob(f"{test_dir}/*.npz")
        test_ds = MedicalDataset(test_paths, test=True)
    else:
        test_ds = None 

    loader = [train_ds, val_ds, test_ds]

    return loader

def get_train_val_test_loader_seperate(train_dir, val_dir, test_dir=None):
    train_datalist = glob.glob(f"{train_dir}/*.npz")
    val_datalist = glob.glob(f"{val_dir}/*.npz")

    print(f"training data is {len(train_datalist)}")
    print(f"validation data is {len(val_datalist)}")

    if test_dir is not None:
        test_datalist = glob.glob(f"{test_dir}/*.npz")
        print(f"test data is {len(test_datalist)}")
        test_ds = MedicalDataset(test_datalist, test=True)
    else :
        test_ds = None

    train_ds = MedicalDataset(train_datalist)
    val_ds = MedicalDataset(val_datalist)
    
    loader = [train_ds, val_ds, test_ds]

    return loader

def get_train_val_test_loader_from_split_json(data_dir, split_json_file):
    import json 

    with open(split_json_file, "r") as f:

        datalist = json.loads(f.read())

    train_datalist = datalist["train"]
    val_datalist = datalist["validation"]
    test_datalist = datalist["test"]

    def add_pre(datalist):
        for i in range(len(datalist)):
            datalist[i] = os.path.join(data_dir, datalist[i])
    
    add_pre(train_datalist)
    add_pre(val_datalist)
    add_pre(test_datalist)
    print(f"training data is {len(train_datalist)}")
    print(f"validation data is {len(val_datalist)}")
    print(f"test data is {len(test_datalist)}", sorted(test_datalist))

    train_ds = MedicalDataset(train_datalist)
    val_ds = MedicalDataset(val_datalist)
    test_ds = MedicalDataset(test_datalist)

    loader = [train_ds, val_ds, test_ds]

    return loader


# def get_train_val_test_loader_from_train(data_dir, train_rate=0.7, val_rate=0.1, test_rate=0.2, seed=46):
#     ## train all labeled data
#     ## fold denote the validation data in training data
#     all_paths = glob.glob(f"{data_dir}/*.npz")
#     # fold_data = get_kfold_data(all_paths, 5)[fold]
#
#     train_number = int(len(all_paths) * train_rate)
#     val_number = int(len(all_paths) * val_rate)
#     test_number = int(len(all_paths) * test_rate)
#     random.seed(seed)
#     # random_state = random.random
#     random.shuffle(all_paths)
#
#     train_datalist = all_paths[:train_number]
#     val_datalist = all_paths[train_number: train_number + val_number]
#     test_datalist = all_paths[-test_number:]
#
#     print(f"training data is {len(train_datalist)}")
#     print(f"validation data is {len(val_datalist)}")
#     print(f"test data is {len(test_datalist)}", sorted(test_datalist))
#
#     train_ds = MedicalDataset(train_datalist)
#     val_ds = MedicalDataset(val_datalist)
#     test_ds = MedicalDataset(test_datalist)
#
#     loader = [train_ds, val_ds, test_ds]
#
#     return loader
# def get_train_val_test_loader_from_train(data_dir, train_rate=0.93, val_rate=0, test_rate=0.07, seed=46, move_from_val_to_test_list=None):
def get_train_val_test_loader_from_train(data_dir, train_rate=0.87, val_rate=0, test_rate=0.13, seed=44,
                                         move_from_val_to_test_list=None, additional_test_data_dir=None):
    ## train all labeled data
    all_paths = glob.glob(f"{data_dir}/*.npz")

    train_number = int(len(all_paths) * train_rate)
    val_number = int(len(all_paths) * val_rate)
    test_number = int(len(all_paths) * test_rate)
    random.seed(seed)

    random.shuffle(all_paths)

    train_datalist = all_paths[:train_number]
    val_datalist = all_paths[train_number: train_number + val_number]
    test_datalist = all_paths[-test_number:]

    # 如果指定了从验证集移动到测试集的文件列表
    if move_from_val_to_test_list:
        for data_name in move_from_val_to_test_list:
            if data_name in val_datalist:
                val_datalist.remove(data_name)
                train_datalist.append(data_name)
            if data_name in test_datalist:
                test_datalist.remove(data_name)
                train_datalist.append(data_name)
            if data_name not in val_datalist and data_name not in test_datalist:
                print(f"{data_name} 不在验证集或测试集中，无法移动到训练集。")

    print(f"training data is========= {len(train_datalist)}")
    print(f"validation data is {len(val_datalist)}")
    print(f"test data is {len(test_datalist)}", sorted(test_datalist))

    # 添加额外的测试数据，如果提供了额外的测试集目录
    if additional_test_data_dir:
        additional_test_paths = glob.glob(f"{additional_test_data_dir}/*.npz")
        print(f"Found {len(additional_test_paths)} additional test files.")
        val_datalist.extend(additional_test_paths)

    # 添加额外的验证数据，类似于测试数据的处理
    # 你也可以对 val 数据做类似的操作，假如你有类似的需求

    train_ds = MedicalDataset(train_datalist)
    val_ds = MedicalDataset(val_datalist)  # 用修改后的验证集
    test_ds = MedicalDataset(test_datalist)  # 用修改后的测试集

    loader = [train_ds, val_ds, test_ds]

    return loader
# def get_train_val_test_loader_from_train(data_dir, train_rate=0.87, val_rate=0, test_rate=0.13, seed=44, move_from_val_to_test_list=None):
#     ## train all labeled data
#     ## fold denote the validation data in training data
#     all_paths = glob.glob(f"{data_dir}/*.npz")
#     # fold_data = get_kfold_data(all_paths, 5)[fold]
#
#     train_number = int(len(all_paths) * train_rate)
#     val_number = int(len(all_paths) * val_rate)
#     test_number = int(len(all_paths) * test_rate)
#     random.seed(seed)
#     # random_state = random.random
#     random.shuffle(all_paths)
#
#     train_datalist = all_paths[:train_number]
#     val_datalist = all_paths[train_number: train_number + val_number]
#     test_datalist = all_paths[-test_number:]
#     # print(train_datalist)
#     # print(train_datalist)
#     # 如果指定了从验证集移动到测试集的文件列表
#     # if move_from_val_to_test_list:
#     #     for data_name in move_from_val_to_test_list:
#     #         if data_name in train_datalist:
#     #             train_datalist.remove(data_name)
#     #             val_datalist.append(data_name)
#     #             test_datalist.append(data_name)
#     #         else:
#     #             print(f"{data_name} 不在训练集中，无法移动到测试集。")
#     if move_from_val_to_test_list:
#         for data_name in move_from_val_to_test_list:
#             # 从验证集中移除并添加到训练集中
#             if data_name in val_datalist:
#                 val_datalist.remove(data_name)
#                 train_datalist.append(data_name)
#
#             # 从测试集中移除并添加到训练集中
#             if data_name in test_datalist:
#                 test_datalist.remove(data_name)
#                 train_datalist.append(data_name)
#
#             # 如果数据不在验证集或测试集中，则打印提示信息
#             if data_name not in val_datalist and data_name not in test_datalist:
#                 print(f"{data_name} 不在验证集或测试集中，无法移动到训练集。")
#     print(f"training data is========= {len(train_datalist)}")
#     print(f"validation data is {len(val_datalist)}")
#     print(f"test data is {len(test_datalist)}", sorted(test_datalist))
#
#     train_ds = MedicalDataset(train_datalist)
#     val_ds = MedicalDataset(test_datalist)
#     test_ds = MedicalDataset(test_datalist)
#
#     loader = [train_ds, val_ds, test_ds]
#
#     return loader
def get_test_loader_from_train(data_dir, train_rate=0, val_rate=0, test_rate=1, seed=46):
    ## train all labeled data
    ## fold denote the validation data in training data
    all_paths = glob.glob(f"{data_dir}/*.npz")
    # fold_data = get_kfold_data(all_paths, 5)[fold]

    train_number = int(len(all_paths) * train_rate)
    val_number = int(len(all_paths) * val_rate)
    test_number = int(len(all_paths) * test_rate)
    random.seed(seed)
    # random_state = random.random
    random.shuffle(all_paths)

    train_datalist = all_paths[:train_number]
    val_datalist = all_paths[train_number: train_number + val_number]
    test_datalist = all_paths[-test_number:]

    print(f"training data is {len(train_datalist)}")
    print(f"validation data is {len(val_datalist)}")
    print(f"test data is {len(test_datalist)}", sorted(test_datalist))

    train_ds = MedicalDataset(train_datalist)
    val_ds = MedicalDataset(test_datalist)
    test_ds = MedicalDataset(test_datalist)

    loader = [train_ds, val_ds, test_ds]

    return loader

def get_train_loader_from_train(data_dir):
    ## train all labeled data 
    ## fold denote the validation data in training data
    all_paths = glob.glob(f"{data_dir}/*.npz")
    # fold_data = get_kfold_data(all_paths, 5)[fold]

    train_ds = MedicalDataset(all_paths)
  
    return train_ds

def get_test_loader_from_test(data_dir):
    all_paths = glob.glob(f"{data_dir}/*.npz")

    test_ds = MedicalDataset(all_paths)
  
    return test_ds

def get_multi_dir_training_loader(data_dir, fold=0, test_dir=None):
    ## train all labeled data 
    ## fold denote the validation data in training data
    all_paths = []
    for p in data_dir:
        paths = glob.glob(f"{p}/*.npz")
        for pp in paths:
            all_paths.append(pp)

    # print(all_paths)
    fold_data = get_kfold_data(all_paths, 5)[fold]

    train_datalist = all_paths
    val_datalist = fold_data["val_data"]  

    print(f"training data is {len(train_datalist)}")
    print(f"validation data is {len(val_datalist)}")
    train_ds = MedicalDataset(train_datalist)
    
    val_ds = MedicalDataset(val_datalist)

    if test_dir is not None:
        test_paths = glob.glob(f"{test_dir}/*.npz")
        test_ds = MedicalDataset(test_paths, test=True)
    else:
        test_ds = None 

    loader = [train_ds, val_ds, test_ds]

    return loader


def get_train_val_test_loader_from_train_muti_modal(data_dir, train_rate=0.80, val_rate=0, test_rate=0.2, seed=44, move_from_val_to_test_list=None):
    ## train all labeled data
    ## fold denote the validation data in training data
    all_paths = glob.glob(f"{data_dir}/*.npz")
    # fold_data = get_kfold_data(all_paths, 5)[fold]

    train_number = int(len(all_paths) * train_rate)
    val_number = int(len(all_paths) * val_rate)
    test_number = int(len(all_paths) * test_rate)
    random.seed(seed)
    # random_state = random.random
    random.shuffle(all_paths)

    train_datalist = all_paths[:train_number]
    val_datalist = all_paths[train_number: train_number + val_number]
    test_datalist = all_paths[-test_number:]
    # print(train_datalist)
    # print(train_datalist)
    # 如果指定了从验证集移动到测试集的文件列表
    if move_from_val_to_test_list:
        for data_name in move_from_val_to_test_list:
            if data_name in train_datalist:
                train_datalist.remove(data_name)
                val_datalist.append(data_name)
                test_datalist.append(data_name)
            else:
                print(f"{data_name} 不在训练集中，无法移动到测试集。")
    print(f"training data is========= {len(train_datalist)}")
    print(f"validation data is {len(val_datalist)}")
    print(f"test data is {len(test_datalist)}", sorted(test_datalist))

    train_ds = MedicalDataset(train_datalist)
    val_ds = MedicalDataset(test_datalist)
    test_ds = MedicalDataset(test_datalist)

    loader = [train_ds, val_ds, test_ds]

    return loader

def get_train_val_test_loader_from_train_new1(data_dir, train_rate=0.7, val_rate=0.1, test_rate=0.2, seed=42):
    ## train all labeled data 
    ## fold denote the validation data in training data
    all_paths = glob.glob(f"{data_dir}/*.npz")
    # fold_data = get_kfold_data(all_paths, 5)[fold]

    train_number = int(len(all_paths) * train_rate)
    val_number = int(len(all_paths) * val_rate)
    test_number = int(len(all_paths) * test_rate)
    random.seed(seed)
    # random_state = random.random
    random.shuffle(all_paths)

    train_datalist = all_paths[:train_number]
    val_datalist = all_paths[train_number: train_number + val_number]
    test_datalist = all_paths[-test_number:] 

    print(f"training data is {len(train_datalist)}")
    print(f"validation data is {len(val_datalist)}")
    print(f"test data is {len(test_datalist)}", sorted(test_datalist))

    train_ds = MedicalDataset(train_datalist)
    val_ds = MedicalDataset(val_datalist)
    test_ds = MedicalDataset(test_datalist)

    loader = [train_ds, val_ds, test_ds]

    return loader
