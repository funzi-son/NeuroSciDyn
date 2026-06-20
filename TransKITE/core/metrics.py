import numpy as np

def compute_valid_time(truth, pred, threshold=2.0):
    """Computes the Valid Time based on a maximum permissible Euclidean error threshold."""
    E = np.linalg.norm(truth - pred, axis=1)
    idx = np.argmax(E >= threshold)
    if E[idx] < threshold:
        return len(truth) 
    return idx

def compute_ade_fde_rmse(truth, pred):
    """Computes standard trajectory displacement errors (ADE, FDE, RMSE)."""
    errors = np.linalg.norm(truth - pred, axis=1)
    rmse = float(np.sqrt(np.mean(errors**2)))
    ade = float(np.mean(errors))
    fde = float(errors[-1])
    return ade, fde, rmse, errors