import torch
import torchvision
from torchvision import transforms

print('Start')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
trans = transforms.Compose([
    transforms.CenterCrop(240),
    transforms.ToTensor(),
    normalize,
])
imagenet_data = torchvision.datasets.ImageFolder('/gpfs/scratch/tomg/data/ILSVRC2012/train',
                                                 transform=trans)
print('Accessing dataset')
a = imagenet_data[0]
print('   A shape = ', a[0].shape)

print('Making loader')
batch_size = 128
data_loader = torch.utils.data.DataLoader(imagenet_data,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=200)
num_batches = len(data_loader)
print(f'Loader has {num_batches} batches')

print('Accessing loader')
batch = next(iter(data_loader))
batch_shape = batch[0].shape
print('   Batch shape = ', batch_shape)
print('   Batch type = ', batch[0].dtype)


print('Loading data')
for i, (x, y) in enumerate(data_loader):
    if i % 10 == 0:
        print(f'\n{i}/{num_batches} = {100.0*i/num_batches}% | ', end='')
    else:
        print('*', end='')

print('Done')

# print('Allocating memory for whole dataset')
# dataset_shape = [batch_shape[0]*num_batches] + list(batch_shape[1:])
# numel = dataset_shape[0]*dataset_shape[1]*dataset_shape[2]*dataset_shape[3]
# print('   Dataset shape = ', dataset_shape)
# print('   Numel = ', numel)
# print('   Gb = ', numel/1e9)
#
# half_dataset_shape = [batch_shape[0]*num_batches//2] + list(batch_shape[1:])
# dataset1 = torch.ByteTensor(*half_dataset_shape)
# #dataset2 = torch.ByteTensor(*half_dataset_shape)
#
# print('Loading data')
# for i, (x,y) in enumerate(data_loader):
#     if i<num_batches/2:
#         start = i*batch_size
#         end = start+batch_size
#         dataset1[start:end,:,:,:] = x
#     else:
#         j = i - num_batches / 2
#         start = j*batch_size
#         end = start+batch_size
#         dataset2[start:end,:,:,:] = x
#     # print progress
#     print(f'\n{i}/{num_batches} = {100.0 * i / num_batches}%', end='')
#     # if i%100 == 0:
#     #     print(f'\n{i}/{num_batches} = {100.0*i/num_batches}%', end='')
#     # else:
#     #     print('*', end='')
#
# print('Done')
