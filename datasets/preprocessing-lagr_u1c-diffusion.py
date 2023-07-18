import h5py
import numpy as np

with h5py.File('Lagr_u3c_diffusion.h5', 'r') as h5f:
    rx0 = np.array(h5f.get('min'))
    rx1 = np.array(h5f.get('max'))
    u3c = np.array(h5f.get('train'))

x_train = (u3c+1)*(rx1-rx0)/2 + rx0
x_train = np.concatenate((x_train[..., :1], x_train[..., 1:2], x_train[..., -1:]))

rx0, rx1 = np.amin(x_train), np.amax(x_train)
x_train = 2*(x_train-rx0)/(rx1-rx0) - 1

with h5py.File('Lagr_u1c_diffusion.h5', 'w') as hf:
    hf.create_dataset('min', data=rx0)
    hf.create_dataset('max', data=rx1)
    hf.create_dataset('train', data=x_train)
