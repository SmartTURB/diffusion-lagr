import numpy as np
from scipy.spatial import distance

def comput_jsd(y_true, y_pred, bins):
    y_min = min(y_true.min(), y_pred.min())
    y_max = max(y_true.max(), y_pred.max())
    hist_true, bin_edges = np.histogram(y_true, bins=bins, range=(y_min, y_max))
    hist_pred, bin_edges = np.histogram(y_pred, bins=bins, range=(y_min, y_max))
    return distance.jensenshannon(hist_true, hist_pred, 2.0)**2

def struct_func(p, dt, u):
    du_p = (u[:, dt:] - u[:, :-dt]) ** p
    return np.mean(du_p), np.std(du_p)

def comput_batch_mean_err(Sp_batch):
    Sp_mean = np.mean(Sp_batch, axis=0)
    Sp_min  = np.amin(Sp_batch, axis=0)
    Sp_max  = np.amax(Sp_batch, axis=0)
    return Sp_mean, np.vstack([Sp_mean - Sp_min, Sp_max - Sp_mean])

def corr_func(dt, u):
    return np.mean(u[:, :-dt] * u[:, dt:])