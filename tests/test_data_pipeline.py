"""Implement a test of the data processing pipeline for ImageNet experiments.

Run this script to get an estimate of ImageNet runtime on your setup.
Tweak parameters in forest/consts.py according to the results of this benchmark.

"""

import torch
import torchvision

import datetime
import time

from forest.data.datasets import ImageNet
from forest.data.cached_dataset import CachedDataset
from forest.consts import imagenet_mean, imagenet_std

from torchvision import transforms

import argparse

parser = argparse.ArgumentParser(description='Construct poisoned training data for the given network and dataset')
parser.add_argument('--data_path', default='/gpfs/scratch/tomg/data/ILSVRC2012/', type=str)
parser.add_argument('--bs', default=128, type=int)
parser.add_argument('--num_batches', default=10, type=int, help='Test for this many batches of training.')

parser.add_argument('--lmdb_path', default='/gpfs/scratch/tomg/lmdb_storage', type=str, help='Database location.')

args = parser.parse_args()

""" PARAMETERS"""
data_path = args.data_path
num_batches = args.num_batches
batch_size = args.bs
imagenet_size = 1_281_167
""" --------- """


transform_minimal = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor()])

transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)])



def test(dataset, workers, shuffle=False, pin_memory=False):
    # print('Construct dataloader:')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                             num_workers=workers, pin_memory=pin_memory)
    # print('Dataloader constructed.')
    # print('Loading epoch ...')
    start_time = time.time()
    for idx, data in enumerate(dataloader):
        if idx % 10 == 0:
            print(f'\n{idx}/{len(dataloader)} = {100.0*idx/len(dataloader)}% | ', end='')
        else:
            print('*', end='')
        # if idx > args.num_batches:
        #    break

    epoch_estimate = (time.time() - start_time)  # * len(dataloader) / args.num_batches
    print(f'Data loading per epoch estimated to be {datetime.timedelta(seconds=epoch_estimate)}.')





if __name__ == "__main__":
    print('Currently evaluating .... Dataloader variants... :')

    print('Testing Torchvision ImageFolder:')
    dataset = torchvision.datasets.ImageFolder(args.data_path + 'train',
                                               transform=transform_minimal)
    test(dataset, workers=200, shuffle=True)

    print('Testing our ImageFolder subclass:')
    dataset = ImageNet(args.data_path,
                       transform=transform_minimal)
    test(dataset, workers=200, shuffle=True)


    print('Testing our ImageFolder subclass without shuffling:')
    dataset = ImageNet(args.data_path,
                       transform=transform_minimal)
    test(dataset, workers=200, shuffle=False)


    print('Testing ours with our data augmentation:')
    dataset = ImageNet(args.data_path,
                       transform=transform_train)
    test(dataset, workers=200, shuffle=True)


    print('Testing ours with pinning:')
    dataset = ImageNet(args.data_path,
                       transform=transform_train)
    test(dataset, workers=200, shuffle=True, pin_memory=True)


    print('Reduce to 40 workers:')
    dataset = ImageNet(args.data_path,
                       transform=transform_train)
    test(dataset, workers=40, shuffle=True, pin_memory=False)


    print('Test LMDB')
    dataset = ImageNet(args.data_path,
                       transform=transform_train)
    dataset = LMDBDataset(dataset, args.lmdb_path, 'train', rebuild_cache=False)
    test(dataset, workers=200, shuffle=True, pin_memory=False)

    print('Test LMDB with memory pinning')
    dataset = ImageNet(args.data_path,
                       transform=transform_train)
    dataset = LMDBDataset(dataset, args.lmdb_path, 'train', rebuild_cache=False)
    test(dataset, workers=200, shuffle=True, pin_memory=True)

    print('Test LMDB 40 workers')
    dataset = ImageNet(args.data_path,
                       transform=transform_train)
    dataset = LMDBDataset(dataset, args.lmdb_path, 'train', rebuild_cache=False)
    test(dataset, workers=40, shuffle=True, pin_memory=True)


    print('Test cached dataset: (0 workers during testing)')
    dataset = ImageNet(args.data_path,
                       transform=transform_train)
    dataset = CachedDataset(dataset, num_workers=200)
    test(dataset, workers=0, shuffle=True, pin_memory=True)

    print('------- Terminating ---------')
