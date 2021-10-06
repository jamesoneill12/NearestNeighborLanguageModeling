from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch


def dataloader(dataset, input_size, batch_size, split='train'):

    scale = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) if 'mnist' not in dataset\
        else transforms.Normalize(mean=[0.5], std=[0.5])
    transform = transforms.Compose([transforms.Resize((input_size, input_size)),
                                    transforms.ToTensor(), scale])

    if dataset == 'mnist':
        data_loader = DataLoader(
            datasets.MNIST('D:/data/mnist', train=True, transform=transform, download=True),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'fashion-mnist':
        data_loader = DataLoader(
            datasets.FashionMNIST('D:/data/fashion-mnist', train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'cifar10':
        data_loader = DataLoader(
            datasets.CIFAR10('D:/data/cifar10', train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'svhn':
        data_loader = DataLoader(
            datasets.SVHN('D:/data/svhn', split=split, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'stl10':
        data_loader = DataLoader(
            datasets.STL10('D:/data/stl10', split=split, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'lsun-bed':
        data_loader = DataLoader(
            datasets.LSUN('D:/data/lsun', classes=['bedroom_train'], transform=transform),
            batch_size=batch_size, shuffle=True)

    elif dataset == 'flickr':
        # flickr_dataset = datasets.ImageFolder(root='D:/data/flickr30k/flickr30k-resized-images/', transform=transform)
        # data_loader = torch.utils.data.DataLoader(flickr_dataset, batch_size=batch_size, shuffle=True, num_workers=4, is_cuda=True)
        flickr_dataset = datasets.Flickr30k(root='D:/data/flickr30k/flickr30k-resized-images/',
                                            ann_file='D:/data/flickr30k/flickr30k_train.json',
                                            transform=transform, target_transform=None)
        data_loader = torch.utils.data.DataLoader(flickr_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    elif dataset == 'coco':
        print("Found COCO, batch size {}".format(batch_size))
        coco_dataset = datasets.CocoCaptions(root='D:/data/coco/images/train2014/resized2014/',
                                annFile='D:/data/coco/annotations/annotations_trainval2014/'
                                        'annotations/captions_train2014.json',
                                transform=transform)
        # coco_dataset = datasets.ImageFolder(root='D:/data/coco/images/train2014/', transform=transform)
        data_loader = torch.utils.data.DataLoader(coco_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    print(dataset)
    return data_loader


if __name__ == "__main__":

    coco_loader = dataloader('coco', input_size=256, batch_size=64)

    for i, sample in enumerate(coco_loader):
        img, cap = sample
        print(img.size())
