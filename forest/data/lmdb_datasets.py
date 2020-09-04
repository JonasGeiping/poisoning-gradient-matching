"""LMBD dataset wrap an existing dataset and create a database if necessary."""

import os
import io
import pickle

import platform
import lmdb

import torch
import torchvision
import numpy as np

class LMDBDataset(torch.utils.data.Dataset):
    """Implement LMDB caching and access.

    Based on https://github.com/pytorch/vision/blob/master/torchvision/datasets/lsun.py
    and
    https://github.com/Lyken17/Efficient-PyTorch/blob/master/tools/folder2lmdb.py
    """

    def __init__(self, dataset, database_path='', name='', rebuild_cache=False):
        """Initialize with a given pytorch dataset."""
        if os.path.isfile(database_path):
            raise ValueError('LMDB path must lead to a folder containing the databases.')
        self.dataset = dataset
        self.path = os.path.join(os.path.expanduser(database_path), f'{type(dataset).__name__}_{name}.lmdb')

        self.img_shape = self.dataset[0][0].shape

        if rebuild_cache:
            if os.path.isfile(self.path):
                os.remove(self.path)
                os.remove(self.path + '-lock')

        # Load or create database
        if os.path.isfile(self.path):
            print(f'Reusing cached database at {self.path}.')
        else:
            os.makedirs(os.path.expanduser(database_path), exist_ok=True)
            print(f'Creating database at {self.path}. This may take some time ...')
            checksum = create_database(self.dataset, self.path, mean=self.data_mean, std=self.data_std)

        # Setup database
        self.db = lmdb.open(self.path, subdir=False, max_readers=128, readonly=True, lock=False,
                            readahead=False, meminit=False, max_spare_txns=128)
        with self.db.begin(write=False) as txn:
            try:
                self.length = pickle.loads(txn.get(b'__len__'))
                self.keys = pickle.loads(txn.get(b'__keys__'))
            except TypeError:
                raise ValueError(f'The provided LMDB dataset at {self.path} is unfinished or damaged.')


    def __getattr__(self, name):
        """Call this only if all attributes of Subset are exhausted."""
        return getattr(self.dataset, name)

    def __len__(self):
        """Draw length from target dataset."""
        return self.length

    def __getitem__(self, index):
        """Get from database. This is still unordered access.

        Cursor access would be a major hassle in conjunction with random shuffles.
        """
        img, target, idx = None, None, None
        with self.db.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])

        # buffer magic
        buffer = io.BytesIO()
        buffer.write(byteflow)
        buffer.seek(0)

        np_img = np.reshape(np.frombuffer(buffer.read(), dtype=np.uint8), self.img_shape)
        img = torch.as_tensor(np_img, dtype=torch.float) / 255
        img = torchvision.transforms.functional.normalize(img, self.data_mean, self.data_std, inplace=True)

        # load label and id
        target, idx = self.dataset.get_target(index)

        return img, target, idx

    def get_target(self, index):
        """Return only the pair (target, idx)."""
        return self.dataset.get_target(index)


def create_database(dataset, database_path, write_frequency=5000, mean=(0, 0, 0), std=(1, 1, 1)):
    """Create an LMDB database from the given pytorch dataset.

    https://github.com/Lyken17/Efficient-PyTorch/blob/master/tools/folder2lmdb.py

    Removed pyarrow dependency
    """
    if platform.system() == 'Linux':
        map_size = 1099511627776 * 2  # Linux can grow memory as needed.
    else:
        raise ValueError('Provide a reasonable default map_size for your operating system.')
    db = lmdb.open(database_path, subdir=False,
                   map_size=map_size, readonly=False,
                   meminit=False, map_async=True)
    txn = db.begin(write=True)
    for idx, (img, target, idx) in enumerate(dataset):
        # unnormalize
        img_uint8 = (_unnormalize(img, mean, std).mul(255).add_(0.5).clamp_(0, 255)).to(torch.uint8)
        # serialize
        byteflow = img_uint8.numpy().tobytes(order='C')
        txn.put(u'{}'.format(idx).encode('ascii'), byteflow)
        if idx % write_frequency == 0:
            print(f"[{idx} / {len(dataset)}]")
            txn.commit()
            txn = db.begin(write=True)

    # finalize dataset
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', pickle.dumps(keys))
        txn.put(b'__len__', pickle.dumps(len(keys)))

def _unnormalize(tensor, mean, std, inplace=False):
    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    tensor.mul_(std[:, None, None]).add_(mean[:, None, None])
    return tensor
