import datetime
import time

import pytorch_ood
import torch
import torchvision.models
import torchvision.transforms as trn
import torchvision.datasets as dset
from pytorch_ood.dataset.img import TinyImages300k
import os
import telegram
from pytorch_ood.model import WideResNet
from pytorch_ood.utils import ToRGB
from torchvision.models import resnet101
from torchvision.utils import save_image
from torch.utils.data import random_split


def get_transforms(dataset_name, img_size=32):
    resize = trn.Resize((img_size, img_size))

    if dataset_name == "cifar10":
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    elif dataset_name == "cifar100":
        mean = [0.5071, 0.4865, 0.4409]
        std = [0.2673, 0.2564, 0.2762]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    transform = trn.Compose([
        trn.RandomHorizontalFlip(),
        trn.RandomCrop(img_size, padding=4),
        trn.ToTensor(),
        trn.Normalize(mean, std)
    ])

    test_transform = trn.Compose([
        trn.ToTensor(),
        trn.Normalize(mean, std),
        resize
    ])

    return transform, test_transform


def get_training_datasets(args):
    """
    This function returns the training datasets for the in-distribution and the out-of-distribution data.

    :param args: the arguments passed to the main function
    :return: the training datasets for the in-distribution and the out-of-distribution data
    """
    # check if inlier dataset is supported
    if args.dataset_in not in ["cifar10", "cifar100"]:
        raise ValueError(f"Unsupported dataset: {args.dataset_in}")

    # check if outlier dataset is supported
    if args.dataset_out not in ["300K", "GAN_IMG"]:
        raise ValueError(f"Unsupported dataset: {args.dataset_out}")

    # check for dataset_in to prepare inliers
    if args.dataset_in == 'cifar10':
        train_transform, _ = get_transforms("cifar10",32)
        train_data_in = dset.CIFAR10('datasets', train=True, transform=train_transform, download=True)

    if args.dataset_in == "cifar100":
        train_transform, _ = get_transforms("cifar100",32)

        train_data_in = dset.CIFAR100('datasets', train=True, transform=train_transform, download=True)

    # check for dataset_out to prepare outliers
    if args.dataset_out == '300K':
        train_transform, _ = get_transforms("cifar10",32)
        train_data_out = TinyImages300k(root="datasets", transform=train_transform,
                                        target_transform=pytorch_ood.utils.ToUnknown(), download=True)

    if args.dataset_out == "GAN_IMG":
        # print(f"LOG: Loading GAN images for {args.dataset_in}... ")
        if args.dataset_in in ["cifar10", "cifar100"]:
            if args.dataset_in == "cifar10":
                gan_file_name = "samples-c10.pt"
            if args.dataset_in == "cifar100":
                gan_file_name = "samples-c100.pt"

            gan_path = os.path.join(os.getcwd(), 'datasets', gan_file_name)
            #print(gan_path)
            gan_data = torch.load(gan_path)
            # normalize gan images and set the target to -1
            train_data_out = [(el / 255, -1) for el in gan_data]

    return train_data_in, train_data_out


def get_model(args):
    """
    This function returns the model for the in-distribution data.

    :param args: the arguments passed to the main function
    :return: the model for the in-distribution data
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model == "wrn":
        if args.dataset_in == "cifar10":
            model = WideResNet(num_classes=10, pretrained="cifar10-pt").to(device).eval()
        if args.dataset_in == "cifar100":
            model = WideResNet(num_classes=100, pretrained="cifar100-pt").to(device).eval()
    else:
        raise ValueError("This model is not defined and implemented!")

    return model


def check_for_epsilon_order(epoch, args):
    """
    This function checks for the epsilon order and returns the epsilon for the current epoch.

    :param epoch: the current epoch
    :param args: the arguments passed to the main function
    :return: the epsilon for the current epoch
    """
    eps_oe = args.eps_oe
    if args.epsilon_order == "up":
        eps_oe = (args.eps_oe / args.epochs) * (epoch + 1)
    if args.epsilon_order == "down":
        eps_oe = args.eps_oe - (args.eps_oe / args.epochs) * epoch
    if args.epsilon_order == "oscillate":
        if epoch % 2 == 0:
            eps_oe = (args.eps_oe / args.epochs) * (epoch + 1)

        if epoch % 2 == 1:
            eps_oe = args.eps_oe - (args.eps_oe / args.epochs) * epoch
    return eps_oe

