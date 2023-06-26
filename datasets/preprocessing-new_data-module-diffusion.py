import numpy as np
import h5py

filename = '/mnt/petaStor/buzzicotti/LaragianTracers/velocities.npy'
x_train = np.load(filename)[..., :1]

rx0, rx1 = np.amin(x_train), np.amax(x_train)
x_train = 2*(x_train-rx0)/(rx1-rx0) - 1

filename_out = 'Lagr_ux_diffusion.h5'
with h5py.File(filename_out, 'w') as hf:
    hf.create_dataset('min', data=rx0)
    hf.create_dataset('max', data=rx1)
    hf.create_dataset('train', data=x_train)
