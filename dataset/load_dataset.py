import os
import numpy as np
from glob import glob

def load_mcyt_txt(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines[1:]:
        values = [float(v) for v in line.strip().split()]
        if len(values) >= 6:
            data.append([values[0], values[1], values[5]])
    return np.array(data, dtype=np.float32)

def load_biosecurid_txt(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines[1:]:
        values = [float(v) for v in line.strip().split()]
        if len(values) >= 7:
            data.append([values[0], values[1], values[6]])
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

def load_dataset(dataset_name):
    if dataset_name == 'MCYT':
        data_root = './data/MCTY/train'
        load_func = load_mcyt_txt
    elif dataset_name == 'BiosecurID':
        data_root = './data/BiosecurID/train'
        load_func = load_biosecurid_txt
    elif dataset_name == 'e-BioSign DS1':
        data_root = './data/e-BioSign DS1/train'
        load_func = load_ebiosign_txt
    elif dataset_name == 'e-BioSign DS2':
        data_root = './data/e-BioSign DS2/train'
        load_func = load_ebiosign_txt
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    
    sig_dict = {}
    user_dirs = sorted(glob(os.path.join(data_root, 'User*')))
    for user_dir in user_dirs:
        user_name = os.path.basename(user_dir)
        user_id = int(user_name.replace('User', ''))  # 提取数字部分
        sig_dict[user_id] = {True: [], False: []}
        genuine_dir = os.path.join(user_dir, 'genuine')
        if os.path.exists(genuine_dir):
            for filepath in sorted(glob(os.path.join(genuine_dir, '*.txt'))):
                try:
                    sig_dict[user_id][True].append(load_func(filepath))
                except Exception as e:
                    print(f'Error loading {filepath}: {e}')
        forge_dir = os.path.join(user_dir, 'forge')
        if os.path.exists(forge_dir):
            for filepath in sorted(glob(os.path.join(forge_dir, '*.txt'))):
                try:
                    sig_dict[user_id][False].append(load_func(filepath))
                except Exception as e:
                    print(f'Error loading {filepath}: {e}')
    return sig_dict
