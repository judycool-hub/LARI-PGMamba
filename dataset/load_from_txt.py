"""
直接从data文件夹读取数据集
"""
import os
import numpy as np
from glob import glob


def load_mcyt_txt(filepath):
    """读取MCYT格式txt: 第一行点数，后续6列"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines[1:]:
        values = [float(v) for v in line.strip().split()]
        if len(values) >= 6:
            data.append([values[0], values[1], values[5]])  # x, y, pressure
    return np.array(data, dtype=np.float32)


def load_biosecurid_txt(filepath):
    """读取BiosecurID格式txt: 第一行点数，后续7列"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines[1:]:
        values = [float(v) for v in line.strip().split()]
        if len(values) >= 7:
            data.append([values[0], values[1], values[6]])  # x, y, pressure
    return np.array(data, dtype=np.float32)


def load_ebiosign_txt(filepath):
    """读取e-BioSign格式txt: 第一行点数，后续4列 (x, y, timestamp, pressure)"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines[1:]:  # Skip first line (number of points)
        values = [float(v) for v in line.strip().split()]
        if len(values) >= 4:
            data.append([values[0], values[1], values[3]])  # x, y, pressure
    return np.array(data, dtype=np.float32)


def load_dataset_from_folder(data_root, load_func):
    """
    从文件夹加载数据集
    返回格式: sigDict[user_id][True/False] = list of (T,3) arrays
    """
    sig_dict = {}
    user_dirs = sorted(glob(os.path.join(data_root, "User*")))
    
    for user_dir in user_dirs:
        user_id = int(os.path.basename(user_dir).replace("User", ""))
        sig_dict[user_id] = {True: [], False: []}
        
        # 真签名
        genuine_dir = os.path.join(user_dir, "genuine")
        if os.path.exists(genuine_dir):
            for filepath in sorted(glob(os.path.join(genuine_dir, "*.txt"))):
                try:
                    sig_dict[user_id][True].append(load_func(filepath))
                except:
                    pass
        
        # 伪签名
        forge_dir = os.path.join(user_dir, "forge")
        if os.path.exists(forge_dir):
            for filepath in sorted(glob(os.path.join(forge_dir, "*.txt"))):
                try:
                    sig_dict[user_id][False].append(load_func(filepath))
                except:
                    pass
    
    return sig_dict


def load_mcyt():
    """加载MCYT数据集"""
    return load_dataset_from_folder("./data/MCTY/train", load_mcyt_txt)


def load_biosecurid():
    """加载BiosecurID数据集"""
    return load_dataset_from_folder("./data/BiosecurID/train", load_biosecurid_txt)


def load_ebiosign_ds1():
    """加载e-BioSign DS1数据集"""
    return load_dataset_from_folder("./data/e-BioSign DS1/train", load_ebiosign_txt)


def load_ebiosign_ds2():
    """加载e-BioSign DS2数据集"""
    return load_dataset_from_folder("./data/e-BioSign DS2/train", load_ebiosign_txt)
