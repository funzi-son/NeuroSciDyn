import os

UUV_CONFIG = {
    # 1. PATHS & DATA CONFIGURATION
    'data_dir': '../../3DMoTraj_DataSet/3DMoTraj',
    'save_dir': 'experiments/UUV_TransKITE_Full',
    'scenes': [f'3Dscene{i}' for i in range(1, 9)],
    
    # 2. SEQUENCE PARAMETERS
    'seq_len': 8,      
    'pred_len': 12,    
    'stride': 2,       
    'fps': 10.0,       
    'dt': 0.1,         
    
    # 3. MODEL ARCHITECTURE
    'input_dim': 3,               
    'hidden_dim': 64,             
    'num_encoder_layers': 2,      
    'nhead': 4,                   
    'dropout': 0.2,               
    'use_kinematics': True,       
    'use_sg_filter': True,        
    
    # 4. TRAINING HYPERPARAMETERS
    'epochs': 300,                
    'batch_size': 64,             
    'learning_rate': 1e-3,        
    'weight_decay': 1e-4,         
    'patience': 30,               
    
    # LR Scheduler (ReduceLROnPlateau)
    'lr_factor': 0.5,
    'lr_patience': 5,
    'min_lr': 1e-5,
    
    # 5. PHYSICS-INFORMED LOSS WEIGHTS
    'alpha': 0.1,  # Velocity Loss weight
    'beta': 0.1,   # Global Consistency Loss weight
    'gamma': 0.1   # Local-update Euler Loss weight
}

def get_uuv_config():
    """Helper function to return a copy of the config and ensure save directory exists."""
    os.makedirs(UUV_CONFIG['save_dir'], exist_ok=True)
    return UUV_CONFIG.copy()