import torch
import torchvision.transforms as trn
import torchvision.datasets as dset
from pytorch_ood.detector import MaxSoftmax, EnergyBased
from torch.utils.data import Dataset, DataLoader
import pytorch_ood
from pytorch_ood.utils import OODMetrics, ToRGB, ToUnknown
from pytorch_ood.dataset.img import Textures, TinyImageNetCrop, TinyImageNetResize, LSUNCrop, LSUNResize, GaussianNoise, \
    UniformNoise, PixMixDataset, FoolingImages, ImageNetR, ImageNetO, ImageNetA
from tqdm import tqdm

import ssl

# Disabling SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context


def get_test_transform():
    img_size = 32
    resize = trn.transforms.Resize((img_size, img_size))
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    return trn.Compose([pytorch_ood.utils.ToRGB(), trn.ToTensor(), trn.Normalize(mean, std), resize])


def get_target_transform():
    return pytorch_ood.utils.ToUnknown()


def create_known_dataset(dataset_to_test):
    test_transform = get_test_transform()
    if dataset_to_test == "cifar10":
        return dset.CIFAR10('datasets', train=False, transform=test_transform, download=True)
    if dataset_to_test == "cifar100":
        return dset.CIFAR100('datasets', train=False, transform=test_transform, download=True)


def create_unknown_dataset(uk_dataset):
    test_transform = get_test_transform()
    target_transform = get_target_transform()

    dataset_dict = {
        "textures": Textures('datasets', transform=test_transform, target_transform=target_transform, download=True),
        "lsun resize": LSUNResize('datasets', transform=test_transform, target_transform=target_transform,
                                  download=True),
        "lsun crop": LSUNCrop('datasets', transform=test_transform, target_transform=target_transform, download=True),
        "gaussian noise": GaussianNoise(length=500, transform=test_transform, target_transform=target_transform),
        "uniform noise": UniformNoise(length=500, transform=test_transform, target_transform=target_transform),
        "tinyimagenet resize": TinyImageNetResize('datasets', transform=test_transform,
                                                  target_transform=target_transform, download=True),
        "tinyimagenet crop": TinyImageNetCrop('datasets', transform=test_transform, target_transform=target_transform,
                                              download=True),
        "fooling images": FoolingImages('datasets', transform=test_transform, target_transform=target_transform,
                                        download=True)
    }

    return dataset_dict.get(uk_dataset, None)


def evaluate_dataset(test_loader, model, outlier_set):
    print(f"Evaluating {outlier_set}...")
    detec = pytorch_ood.detector.MaxSoftmax(model=model)
    metrics = OODMetrics()
    metrics.reset()

    for x, y in tqdm(test_loader):
        x, y = x.cuda(), y.cuda()
        metrics.update(detec(x), y)

    met = metrics.compute()

    return {
        "dataset": outlier_set,
        "metrics": met
    }


def create_metrics_dataset(model, dataset_to_test):
    model.eval()
    model.cuda()
    dataset_in_test = create_known_dataset(dataset_to_test)

    outlier_sets = [
        "textures",
        "lsun resize",
        "lsun crop",
        "tinyimagenet crop",
        #"tinyimagenet resize",
        #"gaussian noise",
        #"uniform noise"
    ]

    metrics = []
    for uk_dataset in outlier_sets:
        dataset_out_test = create_unknown_dataset(uk_dataset)
        try:
            dataset_test_complete = dataset_in_test + dataset_out_test
        except Exception as e:
            print(f"Error in adding dataset {uk_dataset}:", e)

        test_loader_out = torch.utils.data.DataLoader(
            dataset_test_complete, batch_size=128, shuffle=False,
            num_workers=0, pin_memory=True)

        metrics.append(evaluate_dataset(test_loader_out, model, uk_dataset))

    # Initialize the sum of the metrics
    metrics_sum = {"AUROC": 0, "AUPR-IN": 0, "AUPR-OUT": 0, "FPR95TPR": 0}

    # Sum up all the metrics
    for metric in metrics:
        for key in metrics_sum.keys():
            metrics_sum[key] += metric["metrics"][key]

    # Calculate the averages
    metrics_average = {key: value / len(metrics) for key, value in metrics_sum.items()}

    metrics.append({
        "dataset": "average",
        "metrics": metrics_average
    })
    return metrics


def create_metrics_imagenet(model):
    model.eval()
    model.cuda()
    imagenet_transform = trn.Compose([
        trn.Resize(256),
        trn.CenterCrop(224),
        ToRGB(),
        trn.ToTensor(),
        trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset_in_test = dset.ImageNet(root='/nfs1/kirchhei/imagenet-2012/', split='val', transform=imagenet_transform)

    # ood test sets
    imagenet_r = ImageNetR(root="/nfs1/botschen/datasets/",
                           download=True, transform=imagenet_transform, target_transform=ToUnknown())
    imagenet_r_loader = DataLoader(imagenet_r + dataset_in_test, batch_size=128, num_workers=12)

    imagenet_o = ImageNetO(root="/nfs1/botschen/datasets/",
                           download=True, transform=imagenet_transform, target_transform=ToUnknown())
    imagenet_o_loader = DataLoader(imagenet_o + dataset_in_test, batch_size=128, num_workers=12)

    imagenet_a = ImageNetA(root="/nfs1/botschen/datasets/",
                           download=True, transform=imagenet_transform, target_transform=ToUnknown())
    imagenet_a_loader = DataLoader(imagenet_a + dataset_in_test, batch_size=128, num_workers=12)

    perf_metrics = []

    with torch.no_grad():
        softmax = MaxSoftmax(model)

        ###
        metrics = OODMetrics()

        for x, y in imagenet_a_loader:
            logits = model(x.cuda())
            metrics.update(softmax.score(logits), y)

        m = metrics.compute()
        m.update({
            "Dataset": "ImageNetA",
            "Method": "Softmax"
        })
        perf_metrics.append(m)

        ###
        metrics = OODMetrics()

        for x, y in imagenet_o_loader:
            logits = model(x.cuda())
            metrics.update(softmax.score(logits), y)

        m = metrics.compute()
        m.update({
            "Dataset": "ImageNetO",
            "Method": "Softmax"
        })
        perf_metrics.append(m)

        ###
        metrics = OODMetrics()

        for x, y in imagenet_r_loader:
            logits = model(x.cuda())
            metrics.update(softmax.score(logits), y)

        m = metrics.compute()
        m.update({
            "Dataset": "ImageNetR",
            "Method": "Softmax"
        })
        perf_metrics.append(m)
        print(f"AUROC -> {m['AUROC']:.3%}")
    return perf_metrics
