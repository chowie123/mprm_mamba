import numpy as np 
import os 
from batchgenerators.utilities.file_and_folder_operations import isfile, subfiles
import multiprocessing

# def _convert_to_npy(npz_file: str, unpack_segmentation: bool = True, overwrite_existing: bool = False) -> None:
#     # try:
#     a = np.load(npz_file)  # inexpensive, no compression is done here. This just reads metadata
#     if overwrite_existing or not isfile(npz_file[:-3] + "npy"):
#         np.save(npz_file[:-3] + "npy", a['data'])
#
#     if unpack_segmentation and (overwrite_existing or not isfile(npz_file[:-4] + "_seg.npy")):
#         np.save(npz_file[:-4] + "_seg.npy", a['seg'])


import numpy as np
from os.path import isfile
import os


def _convert_to_npy(npz_file: str, unpack_segmentation: bool = True, overwrite_existing: bool = False) -> None:
    # 目标文件的路径
    # target_npy_path = '/DATA/panxiang/MWL/SegMamba-main/data/fullres/RHUH_NPZ/RHUH-0001.npy'

    # 获取目标文件的数据类型
    # target_dtype = np.load(target_npy_path).dtype

    # 预定义全零数组的形状
    default_shape = (4, 155, 240, 240)

    # 读取 npz 文件
    a = np.load(npz_file)  # 只读取元数据

    # 如果没有 'data' 字段，则创建全0数组
    if 'data' not in a:
        print(f"'{npz_file}' does not contain 'data'. Creating default array with shape {default_shape}.")
        data_array = np.zeros(default_shape, dtype=np.int32)
    else:
        data_array = a['data']

    # 保存 data 到 .npy 文件
    if overwrite_existing or not isfile(npz_file[:-3] + "npy"):
        np.save(npz_file[:-3] + "npy", data_array)

    # 如果 unpack_segmentation 为 True，则保存 seg 到 _seg.npy 文件
    if unpack_segmentation and 'seg' in a and (overwrite_existing or not isfile(npz_file[:-4] + "_seg.npy")):
        np.save(npz_file[:-4] + "_seg.npy", a['seg'])


def unpack_dataset(folder: str, unpack_segmentation: bool = True, overwrite_existing: bool = False,
                   num_processes: int = 8):
    """
    all npz files in this folder belong to the dataset, unpack them all
    """
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        npz_files = subfiles(folder, True, None, ".npz", True)
        p.starmap(_convert_to_npy, zip(npz_files,
                                       [unpack_segmentation] * len(npz_files),
                                       [overwrite_existing] * len(npz_files))
                  )
