import json
import numpy as np
import os
import time
import torch
import torchvision.datasets as dset
from omegaconf import OmegaConf
from tqdm import tqdm
from torch.utils.data import Dataset
from pytorch_ood.loss import OutlierExposureLoss
from pathlib import Path
from attacks import fgsm_softmax, pgd_softmax, mifgsm_softmax
from eval import create_metrics_dataset,create_metrics_imagenet
from utils import get_transforms, get_training_datasets, get_model, check_for_epsilon_order, save_altered_images

from logic.utility.telegram import send_telegram_message_finish, send_telegram_message_start
from noise import gaussian_noise, salt_pepper, moire


def generate_adversarial_outlier(args, model, inlier, outlier, target_inlier, target_outlier, eps_oe, device):
    attack_methods = {
        "None": lambda *args, **kwargs: outlier,
        "FGSM": fgsm_softmax,
        "PGD": pgd_softmax,
        "MIFGSM": mifgsm_softmax
    }

    if args.score == "softmax" and args.adv_oe in attack_methods:
        attack_method = attack_methods[args.adv_oe]
        common_kwargs = {
            "model": model,
            "inlier": inlier,
            "outlier": outlier,
            "target_inlier": target_inlier,
            "target_outlier": target_outlier,
            "eps_oe": eps_oe,
            "device": device
        }
        if args.adv_oe in ["PGD", "MIFGSM"]:
            common_kwargs.update({
                "alpha_oe": args.alpha_oe,
                "steps_oe": args.steps_oe
            })
        return attack_method(**common_kwargs)
    else:
        return None


def generate_noise_outlier(args, outlier):
    noise_methods = {
        "None": lambda *args, **kwargs: outlier,
        "gaussian": lambda image_tensor, **kwargs: gaussian_noise(image_tensor=image_tensor, mean=0,
                                                                  var=args.noise_var),
        "saltpepper": lambda image_tensor, **kwargs: salt_pepper(image_tensor=image_tensor, p=args.noise_p),
        "moire": lambda image_tensor, **kwargs: moire(image_tensor=image_tensor, wavelength=args.noise_wavelength,
                                                      amplitude=args.noise_amplitude)
    }
    try:
        if args.noise_type in noise_methods:
            noise_method = noise_methods[args.noise_type]
            return noise_method(image_tensor=outlier)
        else:
            return outlier
    except:
        #print("No noise type provided.")
        return outlier


def get_device():
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU for computations.")
        return torch.device('cuda')
    else:
        print("CUDA is not available. Using CPU for computations.")
        return torch.device('cpu')


def create_data_loader(data, batch_size, shuffle, num_workers=0, pin_memory=True):
    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle,
                                       num_workers=num_workers, pin_memory=pin_memory)


def prepare_data_loaders(args):
    # Retrieve logic data
    train_data_in, train_data_out = get_training_datasets(args)

    # Define data loaders
    train_loader_in = create_data_loader(train_data_in, args.batch_size, shuffle=True)
    train_loader_out = create_data_loader(train_data_out, args.oe_batch_size, shuffle=False)

    return train_loader_in, train_loader_out


def create_optimizer_and_scheduler(model, args, train_loader_in):
    # define logic parameters
    def cosine_annealing(step, total_steps, lr_max, lr_min):
        return lr_min + (lr_max - lr_min) * 0.5 * (
                1 + np.cos(step / total_steps * np.pi))

    optimizer = torch.optim.SGD(
        model.parameters(), args.learning_rate, momentum=args.momentum,
        weight_decay=args.decay, nesterov=True)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            args.epochs * len(train_loader_in),
            1,  # since lr_lambda computes multiplicative factor
            1e-6 / args.learning_rate))

    return optimizer, scheduler


async def run(args):
    torch.manual_seed(1)
    np.random.seed(1)

    # read in arguments
    print(OmegaConf.to_yaml(args))

    # create save path
    architecture_folder = "save"
    ignore_metrics = ["override", "activate_telegram", "telegram_token", "chat_id", "save_image", "decay", "momentum",
                      "test_bs", "oe_batch_size", "batch_size", "learning_rate"]

    for key in args.keys():
        if key not in ignore_metrics:
            architecture_folder = os.path.join(architecture_folder, str(args[key]))
        else:
            print(f"Ignored {key}")

    print(architecture_folder)
    if os.path.isfile(os.path.join(architecture_folder, "results.txt")):
        if not args.override:
            print(
                "This experiment has already been computed! To recompute, execute the configuration with "
                "override=True")
            return
    Path(architecture_folder).mkdir(parents=True, exist_ok=True)

    # create .txt file to log experiment values
    fileName = "parameters"
    file = open(os.path.join(architecture_folder, fileName) + ".txt", "w")
    for key, value in args.items():
        file.write("%s:%s\n" % (key, value))
    file.close()

    await send_telegram_message_start(args)

    # GPU acceleration
    device = get_device()
    print("Using device:", device)

    model = get_model(args)

    # define data loaders
    train_loader_inlier, train_loader_outlier = prepare_data_loaders(args)

    # define optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(model, args, train_loader_inlier)

    criterion = OutlierExposureLoss()

    def train_adversarial_default(epoch):
        loss_avg = 0.0
        model.train()  # enter train mode
        # start at a random point of the outlier dataset; this induces more randomness without obliterating locality
        if args.dataset_out != "GAN_IMG":
            train_loader_outlier.dataset.offset = np.random.randint(len(train_loader_outlier.dataset))
        print("Starting adversarial training epoch %d" % epoch)
        print("Length of train_loader_inlier: ", len(train_loader_inlier))
        print("Length of train_loader_outlier: ", len(train_loader_outlier))

        for in_set, out_set in tqdm(zip(train_loader_inlier, train_loader_outlier), total=len(train_loader_inlier)):

            # define variables
            inlier = in_set[0]
            outlier = out_set[0]
            target_inlier = in_set[1]
            target_outlier = out_set[1]

            # stop if we have covered the entire outlier dataset and we have remains (10000%64=16 -> the code would crash)
            if len(outlier) < args.oe_batch_size:
                break

            # check for custom epsilon order
            eps_oe = check_for_epsilon_order(epoch, args)

            # perform adversarial attack on outliers
            perturbed_outlier = generate_adversarial_outlier(args, model, inlier, outlier, target_inlier,
                                                             target_outlier, eps_oe, device)

            # check for noise experiments
            perturbed_outlier = generate_noise_outlier(args, perturbed_outlier)

            if args.save_image:
                save_altered_images(outlier, perturbed_outlier, epoch, architecture_folder)

            # concatenate inlier and perturbed outlier
            data = torch.cat((inlier.to(device), perturbed_outlier.to(device)), 0).to(device)

            # forward
            data = data.float()
            x = model(data)

            # backward
            optimizer.zero_grad()

            # calculate loss
            loss = criterion(x, torch.cat((target_inlier.to(device), target_outlier.to(device)), 0))

            # update parameters
            loss.backward()
            optimizer.step()

            scheduler.step()
            # exponential moving average
            loss_avg = loss_avg * 0.8 + float(loss) * 0.2
        train_loss = loss_avg

        return train_loss

    def train_adversarial_imagenet(iters):
        loss_avg = 0.0
        model.train()  # enter train mode
        # start at a random point of the outlier dataset; this induces more randomness without obliterating locality
        if args.dataset_out != "GAN_IMG":
            train_loader_outlier.dataset.offset = np.random.randint(len(train_loader_outlier.dataset))
        print("Starting adversarial imagenet training for %d iterations" % iters)
        print("Length of train_loader_inlier: ", len(train_loader_inlier))
        print("Length of train_loader_outlier: ", len(train_loader_outlier))

        it_train_in = iter(train_loader_inlier)
        it_train_out = iter(train_loader_outlier)
        bar = tqdm(range(iters))

        mav = 0.0
        mavs = []
        aurocs = []

        for i in bar:
            try:
                inlier,target_inlier = next(it_train_in)
                outlier,target_outlier = next(it_train_out)
            except:
                it_train_in = iter(train_loader_inlier)
                it_train_out = iter(train_loader_outlier)
                
            # stop if we have covered the entire outlier dataset and we have remains (10000%64=16 -> the code would crash)
            if len(outlier) < args.oe_batch_size:
                continue

            # check for custom epsilon order
            eps_oe = check_for_epsilon_order(0, args)

            # perform adversarial attack on outliers
            perturbed_outlier = generate_adversarial_outlier(args, model, inlier, outlier, target_inlier,
                                                             target_outlier, eps_oe, device)

            # check for noise experiments
            perturbed_outlier = generate_noise_outlier(args, perturbed_outlier)

            if args.save_image:
                save_altered_images(outlier, perturbed_outlier, i, architecture_folder)

            # concatenate inlier and perturbed outlier
            data = torch.cat((inlier.to(device), perturbed_outlier.to(device)), 0).to(device)

            # forward
            data = data.float()
            y_hat = model(data)

            # backward
            optimizer.zero_grad()

            # calculate loss
            loss = criterion(y_hat, torch.cat((target_inlier.to(device), target_outlier.to(device)), 0))

            # update parameters
            loss.backward()
            optimizer.step()

            scheduler.step()
            # exponential moving average
            mav = 0.2 * loss.item() + 0.8 * mav
            mavs.append(mav)

            bar.set_postfix({"loss": mav})
            if i % 100 == 0:
                # auroc = create_metrics_imagenet(model=model)
                auroc = "empty"
                aurocs.append({
                    "loss": mav,
                    "auroc": auroc
                })
                print(f"AUROC for iteration {i}: {auroc}")

        return mavs,aurocs
    def test():
        dataset_mapping_default = {
            "cifar10": dset.CIFAR10,
            "cifar100": dset.CIFAR100,
        }

        if args.dataset_in in dataset_mapping_default:
            Dataset = dataset_mapping_default[args.dataset_in]
            _, test_transform = get_transforms(args.dataset_in)
            dataset = Dataset(root="datasets", train=False, transform=test_transform, download=True)

        elif args.dataset_in == "imagenet":
            _,test_transform = get_transforms("imagenet", 224)
            print(test_transform)
            dataset = dset.ImageNet('/nfs1/kirchhei/imagenet-2012/', split='test', transform=test_transform,
                                          download=False)

        else:
            raise ValueError("Unknown dataset: {}".format(args.dataset_in))

        test_set_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=128,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        model_to_test = model.to(device)
        softmax = torch.nn.Softmax(dim=1)
        correct = 0
        for data, target in test_set_dataloader:
            data, target = data.to(device), target.to(device)

            # forward
            output = softmax(model_to_test(data))

            # accuracy
            preds = output.argmax(dim=1)
            correct += (preds == target).sum().item()

        accuracy = correct / len(test_set_dataloader.dataset)
        print(f"accuracy: {accuracy}")
        print(f"percentage of false predictions: {1 - accuracy}")
        return accuracy

    async def run_training():
        training_start = time.time()
        losses = []

        # Main training loop for default sets
        if args.dataset_in in ["cifar10", "cifar100"]:
            for epoch in range(args.epochs):
                begin_epoch = time.time()

                train_loss_epoch = train_adversarial_default(epoch)
                losses.append({"epoch": epoch, "loss": round(train_loss_epoch, 4)})

                print(
                    f'Epoch {epoch + 1:3d} | Time {int(time.time() - begin_epoch):5d} | Train Loss {train_loss_epoch:.4f}')
        
        # Main training loop for imagenet
        elif args.dataset_in == "imagenet":
            losses,aurocs_train = train_adversarial_imagenet(iters=args.iter)

        else:
            raise ValueError("Unknown dataset: {}".format(args.dataset_in))

        time_needed = time.time() - training_start
        print(f"Finished training in {time_needed}")

        if args.dataset_in == "imagenet":
            aurocs = create_metrics_imagenet(model=model)
        else:
            aurocs = create_metrics_dataset(model=model, dataset_to_test=args.dataset_in)

        print(aurocs)
        results_path = os.path.join(architecture_folder, "results")
        with open(f"{results_path}.txt", 'w') as json_file:
            json.dump(aurocs_train, json_file)



        print(f"Dumped data to {results_path}.txt")

        await send_telegram_message_finish(args=args, results=aurocs, losses=[], time=time_needed)

    # now run the program :)
    await run_training()
