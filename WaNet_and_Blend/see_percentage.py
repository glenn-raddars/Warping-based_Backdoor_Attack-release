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
from utils.dataloader import PostTensorTransform, get_dataloader
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


def train(netC, optimizerC, schedulerC, train_dl, opt):
    print(" Train:")
    netC.train()
    total_loss_ce = 0
    total_sample = 0

    total_clean = 0
    total_clean_correct = 0
    criterion_CE = torch.nn.CrossEntropyLoss()

    transforms = PostTensorTransform(opt).to(opt.device)

    for batch_idx, (inputs, targets) in enumerate(train_dl):
        optimizerC.zero_grad()

        inputs, targets = inputs.to(opt.device), targets.to(opt.device)
        bs = inputs.shape[0]



        total_inputs = transforms(inputs)
        total_preds = netC(total_inputs)

        loss_ce = criterion_CE(total_preds, targets)

        loss = loss_ce
        loss.backward()

        optimizerC.step()

        total_sample += bs
        total_loss_ce += loss_ce.detach()

        total_clean += bs
        total_clean_correct += torch.sum(
            torch.argmax(total_preds, dim=1) == targets
        )

        avg_acc_clean = total_clean_correct * 100.0 / total_clean

        avg_loss_ce = total_loss_ce / total_sample

        progress_bar(
            batch_idx,
            len(train_dl),
            "CE Loss: {:.4f} | Clean Acc: {:.4f} | ".format(avg_loss_ce, avg_acc_clean),
        )

    schedulerC.step()

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



maxiter = 7
eps = 8./255.
alpha = eps / 5
import torch as th
device = 'cuda:0'
def get_adversarial_examples(netC, train_dl):
    adversarial_examples = []
    for batch_idx, (inputs, targets) in enumerate(train_dl):
        inputs, targets = inputs.to(device), targets.to(device)
        netC.eval()
        inputs_orig = inputs.clone().detach()
        inputs.requires_grad = True
        labels = targets
        for iteration in range(maxiter):
            optimx = th.optim.SGD([inputs], lr=1.)
            optim = th.optim.SGD(netC.parameters(), lr=1.)
            optimx.zero_grad()
            optim.zero_grad()
            output = netC(inputs)
            pgd_loss = -1 * th.nn.functional.cross_entropy(output, labels)
            pgd_loss.backward()

            inputs.grad.data.copy_(alpha * th.sign(inputs.grad))
            optimx.step()
            inputs = th.min(inputs, inputs_orig + eps)
            inputs = th.max(inputs, inputs_orig - eps)
            # inputs = th.clamp(inputs, min=0, max=1)
            inputs = inputs.clone().detach()
            inputs.requires_grad = True

        optimx.zero_grad()
        optim.zero_grad()
        inputs.requires_grad = False
        inputs = inputs.clone().detach()
        adversarial_examples.append((inputs, targets))
    return adversarial_examples



def all2all_eval(netC, test_dl, opt):
    netC.eval()
    correct = 0
    for batch_idx, (inputs, targets) in enumerate(test_dl):
        with torch.no_grad():
            inputs, targets = inputs.to(device), targets.to(device)
            preds_clean = netC(inputs)
            targets_bd = torch.remainder(targets + 1, opt.num_classes)
            correct += torch.sum(torch.argmax(preds_clean, 1) == targets_bd).item()
    print(correct)
    return correct

def all2all_cm(netC, test_dl, opt):
    netC.eval()
    correct = np.zeros((10,10))
    for batch_idx, (inputs, targets) in enumerate(test_dl):
        with torch.no_grad():
            inputs, targets = inputs.to(device), targets.to(device)
            preds_clean = netC(inputs)
            pred = torch.argmax(preds_clean, 1)

            for i in range(len(inputs)):
                correct[(targets[i]+1)%10, pred[i]] += 1

            # targets_bd = torch.remainder(targets + 1, opt.num_classes)
            # correct += torch.sum(torch.argmax(preds_clean, 1) == targets_bd).item()
    print(correct)
    return correct

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

    test_clean_dl = get_dataloader(opt, False)
    # test_bd_dl = get_dataloader(opt, False, inject_portion=1)

    # prepare model
    netC, optimizerC, schedulerC = get_model(opt)


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

    eval(netC, None, None, test_clean_dl, noise_grid, identity_grid, opt)

    # best_clean_acc, best_bd_acc = eval(
    #     netC,
    #     test_clean_dl,
    #     test_bd_dl,
    #     0,
    #     0,
    #     opt,
    # )

    pred = all2all_eval(netC, test_clean_dl, opt)
    exs = get_adversarial_examples(netC, test_clean_dl)
    pred = all2all_eval(netC, exs, opt)
    pred = all2all_cm(netC, exs, opt)
    # with open('clean_cm.txt', 'w') as f:
    #     for i in range(10):
    #         for j in range(10):
    #             f.write(str(int(pred[i][j])) + ',')
    #         f.write('\n')



if __name__ == "__main__":
    main()
