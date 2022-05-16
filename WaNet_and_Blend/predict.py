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
import matplotlib.pyplot as plt
import cv2
#from utils.utils import progress_bar


class opt:
    def __init__(self):
        self.dataset = "mnist"
        self.data_root = "./data"
        self.temps = "./temps"
        self.device = "cuda"
        self.continue_training = False
        self.attack_mode = "all2one"
        self.bs = 128
        self.lr_C = 1e-2
        self.schedulerC_milestones = [100, 200, 300, 400]
        self.schedulerC_lambda = 0.1
        self.n_iters = 600
        self.num_workers = 6
        self.target_label = 0
        self.pc = 0.1
        self.cross_ratio = 2 # rho_a = pc, rho_n = pc * cross_ratio
        self.random_rotation = 10
        self.random_crop = 5
        self.s = 0.5
        self.k = 4
        self.grid_rescale = 1 # scale grid values to avoid pixel values going out of [-1, 1]. For example, grid-rescale = 0.98
        self.checkpoints = "./all2all_part_Blending"

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

            # 只是单纯的每个人加上blend后门小点
            inputs_Blend = torch.tensor(inputs)
            patten = cv2.resize(cv2.imread('./img/dot.jpg', 0), (28, 28)).reshape(1, 28, 28).astype(np.float32) / 255
            patten = torch.tensor(patten)
            patten = patten.to(opt.device)
            for i in range(inputs.shape[0]):
                inputs_Blend[i][0] = inputs[i][0]*(1 - 0.2) + patten*0.2

            preds_Blend = netC(inputs_Blend)


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
            print()

            # 在WaNet的基础上在加上blend后门
            inputs_WaNet_Blend = torch.tensor(inputs_bd)
            patten = cv2.resize(cv2.imread('./img/dot.jpg', 0), (28, 28)).reshape(1, 28, 28).astype(np.float32) / 255
            patten = torch.tensor(patten)
            patten = patten.to(opt.device)
            for i in range(inputs.shape[0]):
                inputs_WaNet_Blend[i][0] = inputs_bd[i][0]*(1 - 0.2) + patten*0.2

            preds_WaNet_Blend = netC(inputs_WaNet_Blend)

            for i in range(25):
                img_clean = inputs[i]
                img_clean = img_clean.resize(28, 28)
                img_clean = img_clean.data.cpu().numpy()#这就变成了一个plt可读文件
                predicted_clean = int(torch.argmax(preds_clean[i], 0))
                tureLabel = int(targets[i])

                img_bd = inputs_bd[i]
                img_bd = img_bd.resize(28, 28)
                img_bd = img_bd.data.cpu().numpy()#这就变成了一个plt可读文件
                predicted_bd = int(torch.argmax(preds_bd[i], 0))
                tureLabel_bd = int(targets_bd[i])

                img_BlD = inputs_Blend[i]
                img_BlD = img_BlD.resize(28, 28)
                img_BlD = img_BlD.data.cpu().numpy()
                predicted_BlD = int(torch.argmax(preds_Blend[i], 0))
                tureLabel_BlD = int(targets[i])

                img_W_B = inputs_WaNet_Blend[i]
                img_W_B = img_W_B.resize(28, 28)
                img_W_B = img_W_B.data.cpu().numpy()
                predicted_W_B = int(torch.argmax(preds_WaNet_Blend[i], 0))
                tureLabel_W_B = int((targets_bd[i]+1) % 10)

                #只是单纯的

                axs = plt.figure().subplots(2, 2)
                axs[0][0].imshow(img_clean)
                axs[0][0].set_title("picture is "+str(predicted_clean) + " ture label is " + str(tureLabel))

                axs[0][1].imshow(img_bd)
                axs[0][1].set_title("picture is "+str(predicted_bd) + " ture label is " + str(tureLabel_bd))

                axs[1][0].imshow(img_BlD)
                axs[1][0].set_title("picture is "+str(predicted_BlD) + " ture label is " + str(tureLabel_BlD))

                axs[1][1].imshow(img_W_B)
                axs[1][1].set_title("picture is "+str(predicted_W_B) + " ture label is " + str(tureLabel_W_B))

                plt.show()

                

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
            # progress_bar(batch_idx, len(test_dl), info_string)


def main():
    global opt
    opt = opt()
    opt.attack_mode = "all2all"

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

    print(opt.ckpt_path)

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


if __name__ == "__main__":
    main()
