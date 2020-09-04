"""Implement a sanity check file that trains on poisoned data independent of any other code.

This file can be run directly as 'python sanity_check.py'.

It will consume images stored under poisons/train, poisons/test and poisons/targets,
retrain a ResNet18 and check if the targets are sucessfully classified with their intended (!) class.

To construct poisons/train, poisons/test, poisons/targets, run the usual framework, i.e. in the easiest case run
python brew_poison.py --net ResNet18 --save full

"""

import torch
import torchvision

from torchvision import transforms
from collections import defaultdict

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
dtype = torch.float
setup = dict(device=device, dtype=dtype)
normalize = True
batch_size = 128
shuffle = True
PIN_MEMORY = False
NON_BLOCKING = False
epochs = 40
weight_decay = 5e-4

data_mean = [0.4914672374725342, 0.4822617471218109, 0.4467701315879822]
data_std = [0.24703224003314972, 0.24348513782024384, 0.26158785820007324]

def main():
    """Execute full routine."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])

    transform_valid = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x : x)])

    trainset = torchvision.datasets.ImageFolder('poisons/train', transform=transform_train)
    validset = torchvision.datasets.ImageFolder('poisons/test', transform=transform_valid)
    targetset = torchvision.datasets.ImageFolder('poisons/targets', transform=transform_valid)

    num_workers = torch.get_num_threads() if torch.get_num_threads() > 1 else 0
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=min(batch_size, len(trainset)),
                                              shuffle=shuffle, drop_last=True, num_workers=num_workers,
                                              pin_memory=PIN_MEMORY)
    validloader = torch.utils.data.DataLoader(validset, batch_size=min(batch_size, len(validset)),
                                              shuffle=False, drop_last=False, num_workers=num_workers,
                                              pin_memory=PIN_MEMORY)
    targetloader = torch.utils.data.DataLoader(targetset, batch_size=min(batch_size, len(targetset)),
                                               shuffle=False, drop_last=False, num_workers=num_workers,
                                               pin_memory=PIN_MEMORY)

    class ResNetCifar(torchvision.models.ResNet):
        """ResNet variant that is better suited to CIFAR-10 (not the 1-to-1 CIFAR-10 from the paper though)."""

        def __init__(self, block, num_blocks, in_channels=3, num_channels=10, **kwargs):
            """Init with args Block, num_blocks, num_classes. Get the block from torchvision.models.resnet.BasicBlock."""
            super().__init__(block, num_blocks, num_channels, **kwargs)

            # Reduce size of conv1, remove first maxpool
            self.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.maxpool = torch.nn.Identity()

    model = model = ResNetCifar(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2])
    model.to(**setup)
    # Define training routine
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs // 2.667, epochs // 1.6,
                                                                            epochs // 1.142], gamma=0.1)

    print('Training started...')
    stats = defaultdict()

    for epoch in range(epochs):

        model.train()
        epoch_loss, total_loss, total_preds, correct_preds = 0, 0, 0, 0
        data = []
        # with torch.autograd.set_detect_anomaly(True):
        for batch, (inputs, labels) in enumerate(trainloader):
            # Prep Mini-Batch
            model.train()
            optimizer.zero_grad()

            # Transfer to GPU
            inputs = inputs.to(**setup)
            labels = labels.to(dtype=torch.long, device=device)

            # Get loss
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            predictions = torch.argmax(outputs.data, dim=1)
            total_preds += labels.size(0)
            correct_preds += (predictions == labels).sum().item()
            total_loss += loss.item()

        scheduler.step()
        # Validate normal images
        model.eval()
        correct = 0
        total = 0
        loss = 0
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(validloader):
                inputs = inputs.to(**setup)
                targets = targets.to(device=setup['device'], dtype=torch.long)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                loss += criterion(outputs, targets).item()
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        val_acc = correct / total
        val_loss = loss / (i + 1)

        # Validate targets
        correct = 0
        total = 0
        loss = 0
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(targetloader):
                inputs = inputs.to(**setup, non_blocking=NON_BLOCKING)
                targets = targets.to(device=setup['device'], dtype=torch.long)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                loss += criterion(outputs, targets).item()
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        target_acc = correct / total
        target_loss = loss / (i + 1)


        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch: {epoch}| lr: {current_lr:.4f} | '
              f'Train loss is {epoch_loss / (batch + 1):2.4f}, acc: {100 * correct_preds / total_preds:2.2f}% | '
              f'Val loss is {val_loss:2.4f}, acc: {100 * val_acc:2.2f}% | '
              f'Target loss is {target_loss:2.4f}, acc: {100 * target_acc:2.2f}% | ')


if __name__ == "__main__":
    print('Currently evaluating .... Sanity Check 1 ... -------------------------------:')
    main()

    print('-----Training finished------ Terminating sanity check ---------')
