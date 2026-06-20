import os

UAV_CONFIG = {
    # 1. PATHS & DATA CONFIGURATION
    'data_dir': '../Synthetic-UAV-Flight-Trajectories',
    'save_dir': 'experiments/UAV_TransKITE_Full',  # <-- Added missing key
    'train_list': '../path_list_train_files.txt',
    'val_list': '../path_list_val_files.txt',
    'test_list': '../path_list_test_files.txt',
    
    # 2. SEQUENCE PARAMETERS (HORIZON)
    'seq_len': 50,
    'pred_len': 20,
    'stride': 5,            
    'dt': 0.1,              

    # 3. MODEL ARCHITECTURE
    'model_name': 'TransKITE',
    'input_dim': 3,         
    'hidden_dim': 64,
    'num_layers': 2,
    'nhead': 4,
    'dropout': 0.1,

    # 4. TRAINING HYPERPARAMETERS
    'batch_size': 64,
    'epochs': 500,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'patience': 25,
    'input_noise': 0.01,    

    # 5. PREPROCESSING FLAGS
    'use_sg_filter': True,  
    'use_kinematics': True, 

    # 6. PHYSICS-INFORMED LOSS WEIGHTS
    'alpha_phys': 0.1,      
    'alpha_cons': 0.1,      
}

def get_uav_config():
    """Helper function to return a copy of the config and ensure save directory exists."""
    os.makedirs(UAV_CONFIG['save_dir'], exist_ok=True)
    return UAV_CONFIG.copy()