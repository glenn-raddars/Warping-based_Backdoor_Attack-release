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


def create_targets_bd(targets, opt):
    if opt.attack_mode == "all2one":
        bd_targets = torch.ones_like(targets) * opt.target_label
    elif opt.attack_mode == "all2all":
        bd_targets = torch.tensor([(label + 1) % opt.num_classes for label in targets])
    else:
        raise Exception("{} attack mode is not implemented".format(opt.attack_mode))
    return bd_targets.to(opt.device)

blends_path = './blend_imgs'
blends_path = [os.path.join(blends_path, i) for i in os.listdir(blends_path)]
from PIL import Image
import torchvision.transforms as transforms
t = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])])
blends_imgs = [t(Image.open(i).convert('RGB').resize((32, 32))).unsqueeze(0).cuda() for i in blends_path]

r = 0.02

def create_bd(inputs, targets, opt):
    bd_targets = create_targets_bd(targets, opt)
    blend_indexs = np.random.randint(0,len(blends_imgs), (5,))
    bd_inputs = inputs * (1-r*5) + blends_imgs[blend_indexs[0]] * r\
                + blends_imgs[blend_indexs[1]] * r\
                + blends_imgs[blend_indexs[2]] * r\
                + blends_imgs[blend_indexs[3]] * r\
                + blends_imgs[blend_indexs[4]] * r
    return bd_inputs, bd_targets


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


import torch as th
maxiter = 5
eps = 4./255.
alpha = eps / 3
def train(netC, optimizerC, schedulerC, train_dl, opt, adv=False):
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


def eval(
    netC,
    test_dl,
    best_clean_acc,
    best_bd_acc,
    opt,
):
    print(" Eval:")
    netC.eval()

    total_sample = 0
    total_clean_correct = 0
    total_bd_correct = 0


    for batch_idx, (inputs, targets) in enumerate(test_dl):
        with torch.no_grad():
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            bs = inputs.shape[0]
            total_sample += bs

            # Evaluate Clean
            preds_clean = netC(inputs)
            total_clean_correct += torch.sum(torch.argmax(preds_clean, 1) == targets)

            # Evaluate Backdoor
            with torch.no_grad():
                bs = inputs.shape[0]
                inputs_bd, targets_bd = create_bd(inputs, targets, opt)

            preds_bd = netC(inputs_bd)
            total_bd_correct += torch.sum(torch.argmax(preds_bd, 1) == targets_bd)

            acc_clean = total_clean_correct * 100.0 / total_sample
            acc_bd = total_bd_correct * 100.0 / total_sample


            info_string = "Clean Acc: {:.4f} - Best: {:.4f} | Bd Acc: {:.4f} - Best: {:.4f}".format(
                acc_clean, best_clean_acc, acc_bd, best_bd_acc
            )
            progress_bar(batch_idx, len(test_dl), info_string)

    return best_clean_acc, best_bd_acc


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
    train_dl = get_dataloader(opt, True)
    test_dl = get_dataloader(opt, False)

    # prepare model
    netC, optimizerC, schedulerC = get_model(opt)

    netC.load_state_dict(torch.load('mutli-blend.pt'))

    best_clean_acc, best_bd_acc = eval(
        netC,
        test_dl,
        0,
        0,
        opt,
    )

    best_clean_acc, best_bd_acc = 0, 0
    for epoch in range(opt.n_iters):
        print("Epoch {}:".format(epoch + 1))
        if epoch < 2:
            train(netC, optimizerC, schedulerC, train_dl, opt, adv=True)
        else:
            train(netC, optimizerC, schedulerC, train_dl, opt, adv=False)
        best_clean_acc, best_bd_acc = eval(
            netC,
            test_dl,
            best_clean_acc,
            best_bd_acc,
            opt,
        )


if __name__ == "__main__":
    main()