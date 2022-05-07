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

# x = t.ones([1,2,2,2])
# a = F.upsample(x, scale_factor=[2,2,2,2])
# print(x)
# print(a)

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
        self.checkpoints = "./all2all"

"""开始拆解train，意图弄清inputs_bd怎么加blending"""
def  train(netC, optimizerC, schedulerC, train_dl, noise_grid, identity_grid, tf_writer, epoch, opt):
    print(" Train:")
    netC.train()
    rate_bd = opt.pc
    total_loss_ce = 0
    total_sample = 0

    total_clean = 0
    total_bd = 0
    total_cross = 0
    total_clean_correct = 0
    total_bd_correct = 0
    total_cross_correct = 0
    criterion_CE = torch.nn.CrossEntropyLoss()
    criterion_BCE = torch.nn.BCELoss()

    denormalizer = Denormalizer(opt)
    transforms = PostTensorTransform(opt).to(opt.device)
    total_time = 0

    avg_acc_cross = 0

    for batch_idx, (inputs, targets) in enumerate(train_dl):
        optimizerC.zero_grad()

        inputs, targets = inputs.to(opt.device), targets.to(opt.device)
        bs = inputs.shape[0]

        # Create backdoor data
        num_bd = int(bs * rate_bd)
        num_cross = int(num_bd * opt.cross_ratio)
        grid_temps = (identity_grid + opt.s * noise_grid / opt.input_height) * opt.grid_rescale
        grid_temps = torch.clamp(grid_temps, -1, 1)# 将输入张量的每一个元素都逼近到[-1,1]之间

        ins = torch.rand(num_cross, opt.input_height, opt.input_height, 2).to(opt.device) * 2 - 1#这好像也是把他转到[-1,1]之间
        grid_temps2 = grid_temps.repeat(num_cross, 1, 1, 1) + ins / opt.input_height#单纯的在第0维度上进行重复，应该是batch_size
        grid_temps2 = torch.clamp(grid_temps2, -1, 1)# 将输入张量的每一个元素都逼近到[-1,1]之间

        inputs_bd = F.grid_sample(inputs[:num_bd], grid_temps.repeat(num_bd, 1, 1, 1), align_corners=True)#只对指定数量的输入进行网格采样
        if opt.attack_mode == "all2one":
            targets_bd = torch.ones_like(targets[:num_bd]) * opt.target_label
        if opt.attack_mode == "all2all":
            targets_bd = torch.remainder(targets[:num_bd]+1, opt.num_classes)

        inputs_cross = F.grid_sample(inputs[num_bd : (num_bd + num_cross)], grid_temps2, align_corners=True)
        print(inputs_bd.shape)
        patten = cv2.resize(cv2.imread('./img/dot.jpg', 0), (28, 28)).reshape(1, 28, 28).astype(np.float32) / 255
        patten = torch.tensor(patten)
        patten = patten.to(opt.device)
        for i in range(inputs_bd.shape[0]):
            inputs_bd[i][0] = inputs_bd[i][0]*(1 - 0.2) + patten*0.2
        for image in inputs_bd:
            image = image.resize(28, 28)
            image = image.data.cpu().numpy()
            plt.imshow(image)
            plt.show()


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


if __name__ == "__main__":
    # global opt
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
    train_dl = get_dataloader(opt, True)
    test_dl = get_dataloader(opt, False)

    # prepare model
    netC, optimizerC, schedulerC = get_model(opt)

    # Load pretrained model
    mode = opt.attack_mode
    opt.ckpt_folder = os.path.join(opt.checkpoints, opt.dataset)#你选择的是哪种训练方式all2all或者all2one
    opt.ckpt_path = os.path.join(opt.ckpt_folder, "{}_{}_morph.pth.tar".format(opt.dataset, mode))
    opt.log_dir = os.path.join(opt.ckpt_folder, "log_dir")
    if not os.path.exists(opt.log_dir):#此文件是否存在
        os.makedirs(opt.log_dir)#不存在，就创建路径

    else:
        print("Train from scratch!!!")#从零开始训练
        best_clean_acc = 0.0
        best_bd_acc = 0.0
        best_cross_acc = 0.0
        epoch_current = 0

        # Prepare grid
        ins = torch.rand(1, 2, opt.k, opt.k) * 2 - 1
        ins = ins / torch.mean(torch.abs(ins))
        noise_grid = (
            F.upsample(ins, size=opt.input_height, mode="bicubic", align_corners=True)
            .permute(0, 2, 3, 1)
            .to(opt.device)
        )
        array1d = torch.linspace(-1, 1, steps=opt.input_height)
        x, y = torch.meshgrid(array1d, array1d)
        identity_grid = torch.stack((y, x), 2)[None, ...].to(opt.device)

        shutil.rmtree(opt.ckpt_folder, ignore_errors=True)
        os.makedirs(opt.log_dir)
        with open(os.path.join(opt.ckpt_folder, "opt.json"), "w+") as f:
            json.dump(opt.__dict__, f, indent=2)
        tf_writer = SummaryWriter(log_dir=opt.log_dir)


    train(netC, optimizerC, schedulerC, train_dl, noise_grid, identity_grid, tf_writer, 1, opt)