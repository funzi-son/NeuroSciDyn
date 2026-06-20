import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import savgol_filter

def compute_velocity(pos_seq, dt=0.1, use_sg_filter=True):
    """Computes velocity using Savitzky-Golay filter or finite differences."""
    if use_sg_filter and len(pos_seq) >= 11:
        return savgol_filter(pos_seq, window_length=11, polyorder=3, deriv=1, delta=dt, axis=0)
    return np.gradient(pos_seq, dt, axis=0)

def read_path_list(data_dir, txt_path):
    """Reads a text file containing relative paths and returns absolute paths."""
    with open(txt_path, 'r') as f:
        files = [line.strip() for line in f if line.strip()]
    return [os.path.join(data_dir, file) for file in files]

def build_pos_vel_lists(file_paths, config, training=False, pos_max=None, vel_max=None):
    """Processes UAV trajectory datasets and prepares scaled features."""
    dt = config.get('dt', 0.1)
    seq_len = config.get('seq_len', 50)
    pred_len = config.get('pred_len', 20)
    stride = config.get('stride', 5)
    use_sg_filter = config.get('use_sg_filter', True)

    if training:
        pos_list, vel_list = [], []
        for file_path in file_paths:
            df = pd.read_csv(file_path)
            positions = df[['tx', 'ty', 'tz']].values.astype(np.float32)
            velocities = compute_velocity(positions, dt, use_sg_filter)
            pos_list.append(positions)
            vel_list.append(velocities)

        all_pos = np.vstack(pos_list)
        all_vel = np.vstack(vel_list)
        pos_max = float(np.linalg.norm(all_pos, axis=1).max())
        vel_max = float(np.linalg.norm(all_vel, axis=1).max())

    samples = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        pos = df[['tx', 'ty', 'tz']].values.astype(np.float32)
        vel = compute_velocity(pos, dt, use_sg_filter)

        n_pos = pos / pos_max
        n_vel = vel / vel_max
        traj = np.concatenate([n_pos, n_vel], axis=1)

        n = len(traj) - seq_len - pred_len + 1
        for i in range(0, n, stride):
            x = traj[i : i+seq_len]
            y = traj[i+seq_len : i+seq_len+pred_len]
            samples.append((torch.FloatTensor(x), torch.FloatTensor(y)))

    return samples, pos_max, vel_max

class TrajectoryDataset(Dataset):
    """Dataset wrapper for UAV trajectories."""
    def __init__(self, samples):
        self.samples = samples
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        return self.samples[idx]

class TransKITE_UUVDataset(Dataset):
    """Dataset for UUV trajectories incorporating local scaling and Savitzky-Golay filtering."""
    def __init__(self, data_dir, scenes, split, seq_len=8, pred_len=12, stride=2, fps=10.0, use_sg=True):
        self.samples = []
        actual_stride = stride if split == 'train' else seq_len + pred_len
        
        for scene in scenes:
            folder_path = os.path.join(data_dir, scene, split)
            if not os.path.exists(folder_path): 
                continue
                
            for f in glob.glob(os.path.join(folder_path, '*.txt')):
                try:
                    df = pd.read_csv(f, sep=r'\s+', header=None, names=['frame_id', 'track_id', 'x', 'y', 'z'])
                    for _, group in df.groupby('track_id'):
                        group = group.sort_values('frame_id')
                        pos = group[['x', 'y', 'z']].values.astype(np.float32)
                        f_ids = group['frame_id'].values.astype(np.float32)
                        
                        if len(pos) < seq_len + pred_len: 
                            continue
                        
                        for i in range(0, len(pos) - seq_len - pred_len + 1, actual_stride):
                            p_window = pos[i : i + seq_len + pred_len]
                            
                            dt = float(np.mean(np.diff(f_ids[i : i + seq_len + pred_len])) / fps)
                            if dt == 0: dt = 0.1 
                            
                            anchor = p_window[seq_len - 1].copy()
                            p_rel = p_window - anchor
                            local_scale = np.abs(p_rel).max(axis=0)  
                            scale = np.maximum(local_scale, 1e-3)
                            p_norm = p_rel / scale
                            
                            if use_sg and len(p_norm) >= 11:
                                v_norm = savgol_filter(p_norm, window_length=11, polyorder=3, deriv=1, delta=dt, axis=0)
                            else:
                                v_norm = np.gradient(p_norm, dt, axis=0)
                                
                            x_pos = p_norm[:seq_len]
                            x_vel = v_norm[:seq_len]
                            x_kinematic = np.concatenate([x_pos, x_vel], axis=-1)
                            
                            y_pos = p_norm[seq_len:]
                            y_vel = v_norm[seq_len:]
                            
                            self.samples.append({
                                'x': torch.FloatTensor(x_kinematic),       
                                'y_pos': torch.FloatTensor(y_pos),         
                                'y_vel': torch.FloatTensor(y_vel),         
                                'dt': torch.FloatTensor([dt]),             
                                'scale': torch.FloatTensor(scale),         
                                'anchor': torch.FloatTensor(anchor),       
                                'scene': scene                             
                            })
                except Exception: 
                    pass

    def __len__(self): 
        return len(self.samples)
        
    def __getitem__(self, idx): 
        return self.samples[idx]