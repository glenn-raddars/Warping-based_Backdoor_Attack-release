import json
import os
import shutil
from time import time

import config
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from classifier_models import PreActResNet18, ResNet18
from networks.models import Denormalizer, NetC_MNIST, Normalizer
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import RandomErasing
from utils.clean_dataloader import PostTensorTransform, get_dataloader
from utils.utils import progress_bar


def get_model(opt):
    netC = None
    optimizerC = None
    schedulerC = None

    if opt.dataset == "cifar10" or opt.dataset == "gtsrb":
        netC = PreActResNet18(num_classes=opt.num_classes).to(opt.device)
    if opt.dataset == "celeba":
        netC = ResNet18().to(opt.device)
    if opt.dataset == "mnist":
        netC = NetC_MNIST().to(opt.device)

    # Optimizer
    optimizerC = torch.optim.SGD(netC.parameters(), opt.lr_C, momentum=0.9, weight_decay=5e-4)

    # Scheduler
    schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, opt.schedulerC_milestones, opt.schedulerC_lambda)

    return netC, optimizerC, schedulerC


def eval(
    netC,
    optimizerC,
    schedulerC,
    test_dl,
    noise_grid,
    identity_grid,
    opt,
):
    print(" Eval:")
    netC.eval()

    total_sample = 0
    total_clean_correct = 0
    total_bd_correct = 0
    total_cross_correct = 0
    total_ae_loss = 0

    criterion_BCE = torch.nn.BCELoss()

    for batch_idx, (inputs, targets) in enumerate(test_dl):
        with torch.no_grad():
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            bs = inputs.shape[0]
            total_sample += bs

            # Evaluate Clean
            preds_clean = netC(inputs)
            total_clean_correct += torch.sum(torch.argmax(preds_clean, 1) == targets)

            # Evaluate Backdoor
            grid_temps = (identity_grid + opt.s * noise_grid / opt.input_height) * opt.grid_rescale
            grid_temps = torch.clamp(grid_temps, -1, 1)

            ins = torch.rand(bs, opt.input_height, opt.input_height, 2).to(opt.device) * 2 - 1
            grid_temps2 = grid_temps.repeat(bs, 1, 1, 1) + ins / opt.input_height
            grid_temps2 = torch.clamp(grid_temps2, -1, 1)

            inputs_bd = F.grid_sample(inputs, grid_temps.repeat(bs, 1, 1, 1), align_corners=True)
            if opt.attack_mode == "all2one":
                targets_bd = torch.ones_like(targets) * opt.target_label
            if opt.attack_mode == "all2all":
                targets_bd = torch.remainder(targets+1, opt.num_classes)
            preds_bd = netC(inputs_bd)
            total_bd_correct += torch.sum(torch.argmax(preds_bd, 1) == targets_bd)

            acc_clean = total_clean_correct * 100.0 / total_sample
            acc_bd = total_bd_correct * 100.0 / total_sample

            # Evaluate cross
            if opt.cross_ratio:
                inputs_cross = F.grid_sample(inputs, grid_temps2, align_corners=True)
                preds_cross = netC(inputs_cross)
                total_cross_correct += torch.sum(torch.argmax(preds_cross, 1) == targets)

                acc_cross = total_cross_correct * 100.0 / total_sample

                info_string = "Clean Acc: {:.4f} | Bd Acc: {:.4f} | Cross: {:.4f}".format(acc_clean, acc_bd, acc_cross)
            else:
                info_string = "Clean Acc: {:.4f} - Best: {:.4f} | Bd Acc: {:.4f} - Best: {:.4f}".format(
                    acc_clean, best_clean_acc, acc_bd, best_bd_acc
                )
            progress_bar(batch_idx, len(test_dl), info_string)
import torch as th
maxiter = 5
eps = 4./255.
alpha = eps / 3
def train(netC, optimizerC, schedulerC, train_dl, noise_grid, identity_grid, tf_writer, epoch, opt, adv):
    print(" Train:")
    netC.train()
    criterion_CE = torch.nn.CrossEntropyLoss()
    transforms = PostTensorTransform(opt).to(opt.device)

    for batch_idx, (inputs, targets) in enumerate(train_dl):
        optimizerC.zero_grad()

        inputs, targets = inputs.to(opt.device), targets.to(opt.device)
        total_inputs = transforms(inputs)
        total_targets = targets
        if adv:
            netC.eval()
            total_inputs_orig = total_inputs.clone().detach()
            total_inputs.requires_grad = True
            labels = total_targets

            for iteration in range(maxiter):
                optimx = th.optim.SGD([total_inputs], lr=1.)
                optim = th.optim.SGD(netC.parameters(), lr=1.)
                optimx.zero_grad()
                optim.zero_grad()
                output = netC(total_inputs)
                pgd_loss = -1 * th.nn.functional.cross_entropy(output, labels)
                pgd_loss.backward()

                total_inputs.grad.data.copy_(alpha * th.sign(total_inputs.grad))
                optimx.step()
                total_inputs = th.min(total_inputs, total_inputs_orig + eps)
                total_inputs = th.max(total_inputs, total_inputs_orig - eps)
                # total_inputs = th.clamp(total_inputs, min=-1.9895, max=2.1309)
                total_inputs = total_inputs.clone().detach()
                total_inputs.requires_grad = True

            optimx.zero_grad()
            optim.zero_grad()
            total_inputs.requires_grad = False
            total_inputs = total_inputs.clone().detach()
            netC.train()

        total_preds = netC(total_inputs)

        loss_ce = criterion_CE(total_preds, total_targets)

        loss = loss_ce
        loss.backward()

        optimizerC.step()

    schedulerC.step()

def main():
    opt = config.get_arguments().parse_args()

    if opt.dataset in ["mnist", "cifar10"]:
        opt.num_classes = 10
    elif opt.dataset == "gtsrb":
        opt.num_classes = 43
    elif opt.dataset == "celeba":
        opt.num_classes = 8
    else:
        raise Exception("Invalid Dataset")

    if opt.dataset == "cifar10":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "gtsrb":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "mnist":
        opt.input_height = 28
        opt.input_width = 28
        opt.input_channel = 1
    elif opt.dataset == "celeba":
        opt.input_height = 64
        opt.input_width = 64
        opt.input_channel = 3
    else:
        raise Exception("Invalid Dataset")

    # Dataset
    test_dl = get_dataloader(opt, False)

    # prepare model
    netC, optimizerC, schedulerC = get_model(opt)

    # Load pretrained model
    mode = opt.attack_mode
    opt.ckpt_folder = os.path.join(opt.checkpoints, opt.dataset)
    opt.ckpt_path = os.path.join(opt.ckpt_folder, "{}_{}_morph.pth.tar".format(opt.dataset, mode))
    opt.log_dir = os.path.join(opt.ckpt_folder, "log_dir")

    if os.path.exists(opt.ckpt_path):
        state_dict = torch.load(opt.ckpt_path)
        netC.load_state_dict(state_dict["netC"])
        identity_grid = state_dict["identity_grid"]
        noise_grid = state_dict["noise_grid"]
    else:
        print("Pretrained model doesnt exist")
        exit()

    eval(
        netC,
        optimizerC,
        schedulerC,
        test_dl,
        noise_grid,
        identity_grid,
        opt,
    )
    train_dl = get_dataloader(opt, train=True)
    for epoch in range(20):
        print("Epoch {}:".format(epoch + 1))
        if epoch < 2:
            train(netC, optimizerC, schedulerC, train_dl, noise_grid, identity_grid, None, epoch, opt, adv=True)
            print('ok')
        else:
            train(netC, optimizerC, schedulerC, train_dl, noise_grid, identity_grid, None, epoch, opt, adv=False)
        eval(
            netC,
            optimizerC,
            schedulerC,
            test_dl,
            noise_grid,
            identity_grid,
            opt,
        )


if __name__ == "__main__":
    main()
