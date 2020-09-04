"""Write a PyTorch dataset into RAM."""

import torch
from ..consts import PIN_MEMORY


class CachedDataset(torch.utils.data.Dataset):
    """Cache a given dataset."""

    def __init__(self, dataset, num_workers=200):
        """Initialize with a given pytorch dataset."""
        self.dataset = dataset
        self.cache = []
        print('Caching started ...')
        batch_size = min(len(dataset) // max(num_workers, 1), 8192)
        cacheloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                  shuffle=False, drop_last=False, num_workers=num_workers,
                                                  pin_memory=False)

        # Allocate memory:
        self.cache = torch.empty((len(self.dataset), *self.dataset[0][0].shape), pin_memory=PIN_MEMORY)

        pointer = 0
        for data in cacheloader:
            batch_length = data[0].shape[0]
            self.cache[pointer: pointer + batch_length] = data[0]  # assuming the first return value of data is the image sample!
            pointer += batch_length
            print(f"[{pointer} / {len(dataset)}] samples processed.")

        print(f'Dataset sucessfully cached into RAM.')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.cache[index]
        target, index = self.dataset.get_target(index)
        return sample, target, index

    def get_target(self, index):
        return self.dataset.get_target(index)

    def __getattr__(self, name):
        """This is only called if all attributes of Subset are exhausted."""
        return getattr(self.dataset, name)
