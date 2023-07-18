from mpi4py import MPI
import h5py
from torch.utils.data import DataLoader, Dataset
import numpy as np


def load_data(
    *,
    dataset_path,
    dataset_name,
    batch_size,
    class_cond=False,
    deterministic=False,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param dataset_path: a dataset path.
    :param dataset_name: a dataset name.
    :param batch_size: the batch size of each returned pair.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. Not implemented.
    :param deterministic: if True, yield results in a deterministic order.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    with h5py.File(dataset_path, 'r', driver='mpio', comm=MPI.COMM_SELF) as f:
        len_dataset = f[dataset_name].len()

    chunk_size = len_dataset // size
    start_idx  = rank * chunk_size

    dataset = TurbDataset(
        dataset_path, dataset_name, class_cond, start_idx, chunk_size,
    )

    shuffle = True if deterministic else False
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1, drop_last=True
    )

    while True:
        yield from loader


class TurbDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        dataset_name,
        class_cond,
        start_idx,
        chunk_size,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.class_cond = class_cond
        self.start_idx  = start_idx
        self.chunk_size = chunk_size

    def __len__(self):
        return self.chunk_size

    def __getitem__(self, idx):
        idx += self.start_idx

        with h5py.File(self.dataset_path, 'r', driver='mpio', comm=MPI.COMM_SELF) as f:
        #with h5py.File(self.dataset_path, 'r') as f:  # replace the above line with this for demo
            data = f[self.dataset_name][idx].astype(np.float32)
            data = np.moveaxis(data, -1, 0)

            out_dict = {}
            if self.class_cond:
                raise NotImplementedError()
                out_dict["y"] = f[self.dataset_name + '_y'][idx]

        return data, out_dict
