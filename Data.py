import torchvision
import torch

def get_data():
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5])
        ])
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5])
        ])
    )

    # train_data_size = len(train_dataset)
    # test_data_size = len(test_dataset)

    # print(f"Train dataset size: {train_data_size}")
    # print(f"Test dataset size: {test_data_size}")

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        drop_last=True)

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=True,
        drop_last=False)
    
    return train_data_loader, test_data_loader