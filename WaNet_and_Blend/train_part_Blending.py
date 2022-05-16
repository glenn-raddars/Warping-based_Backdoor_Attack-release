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
# from utils.utils import progress_bar

import cv2


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
        # print(bs)

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

        # 给inputs_bd嵌入Blending后门
        patten = cv2.resize(cv2.imread('./img/dot.jpg', 0), (28, 28)).reshape(1, 28, 28).astype(np.float32) / 255
        patten = torch.tensor(patten)
        patten = patten.to(opt.device)
        # print(int(inputs_bd.shape[0]*(3/5)))
        num_W_B = int(inputs_bd.shape[0]*(3/5))#既有WaNet，又有Blending的输入
        for i in range(num_W_B):
            inputs_bd[i][0] = inputs_bd[i][0]*(1 - 0.2) + patten*0.2

        if opt.attack_mode == "all2one":
            targets_W_B = torch.ones_like(targets[:num_W_B]) * opt.target_label
        if opt.attack_mode == "all2all":
            targets_W_B = torch.remainder(targets[:num_W_B]+2, opt.num_classes)# 这个时候就让target_W_B在加2

        total_inputs = torch.cat([inputs_bd, inputs_cross, inputs[(num_bd + num_cross) :]], dim=0)
        total_inputs = transforms(total_inputs)#这就是对图像进行裁剪
        total_targets = torch.cat([targets_W_B ,targets_bd[num_W_B:], targets[num_bd:]], dim=0)# 拼接target
        start = time()
        total_preds = netC(total_inputs)
        total_time += time() - start#训练时间

        loss_ce = criterion_CE(total_preds, total_targets)#交叉熵

        loss = loss_ce
        loss.backward()

        optimizerC.step()

        total_sample += bs#样本数量
        total_loss_ce += loss_ce.detach()#返回的是这一次训练的loss只不过不带梯度

        total_clean += bs - num_bd - num_cross#干净数据集的个数
        total_bd += num_bd#bd投毒数据集的个数
        total_cross += num_cross#cross投毒数据集的个数
        total_clean_correct += torch.sum(
            torch.argmax(total_preds[(num_bd + num_cross) :], dim=1) == total_targets[(num_bd + num_cross) :]
        )#干净数据集预测正确的个数
        wn_pre = torch.sum(torch.argmax(total_preds[num_W_B :num_bd], dim=1) == targets_bd[num_W_B:])#bd后门攻击正确的个数,未加入Blending后门
        wn_bd_pre = torch.sum(torch.argmax(total_preds[:num_W_B], dim=1) == targets_W_B)#后门攻击的正确个数，加入了Blending后门
        all_pre = wn_bd_pre + wn_pre#所有的后门攻击正确数
        total_bd_correct += all_pre#总和


        if num_cross:
            total_cross_correct += torch.sum(
                torch.argmax(total_preds[num_bd : (num_bd + num_cross)], dim=1)
                == total_targets[num_bd : (num_bd + num_cross)]
            )#cross毒化数据集的攻击成功的个数
            avg_acc_cross = total_cross_correct * 100.0 / total_cross

        avg_acc_clean = total_clean_correct * 100.0 / total_clean
        avg_acc_bd = total_bd_correct * 100.0 / total_bd

        avg_loss_ce = total_loss_ce / total_sample


        # if num_cross:
        #     progress_bar(
        #         batch_idx,
        #         len(train_dl),
        #         "CE Loss: {:.4f} | Clean Acc: {:.4f} | Bd Acc: {:.4f} | Cross Acc: {:.4f}".format(
        #             avg_loss_ce, avg_acc_clean, avg_acc_bd, avg_acc_cross
        #         ),
        #     )
        # else:
        #     progress_bar(
        #         batch_idx,
        #         len(train_dl),
        #         "CE Loss: {:.4f} | Clean Acc: {:.4f} | Bd Acc: {:.4f} ".format(avg_loss_ce, avg_acc_clean, avg_acc_bd),
        #     )

        # Save image for debugging
        if not batch_idx % 50:
            if not os.path.exists(opt.temps):
                os.makedirs(opt.temps)
            path = os.path.join(opt.temps, "backdoor_image.png")
            torchvision.utils.save_image(inputs_bd, path, normalize=True)

        # Image for tensorboard
        if batch_idx == len(train_dl) - 2:
            residual = inputs_bd - inputs[:num_bd]
            batch_img = torch.cat([inputs[:num_bd], inputs_bd, total_inputs[:num_bd], residual], dim=2)
            batch_img = denormalizer(batch_img)
            batch_img = F.upsample(batch_img, scale_factor=(4, 4))#将图片的高宽变成原来的4倍
            grid = torchvision.utils.make_grid(batch_img, normalize=True)

    if num_cross:
        print("CE Loss: {:.4f} | Clean Acc: {:.4f} | Bd Acc: {:.4f} | Cross Acc: {:.4f}".format(avg_loss_ce, avg_acc_clean, avg_acc_bd, avg_acc_cross))
    
    else:
        print("CE Loss: {:.4f} | Clean Acc: {:.4f} | Bd Acc: {:.4f} ".format(avg_loss_ce, avg_acc_clean, avg_acc_bd))

    Loss = torch.tensor(avg_loss_ce)
    acc_clean = torch.tensor(avg_acc_clean)
    acc_bd = torch.tensor(avg_acc_bd)
    torch.save(Loss, opt.loss_dir + "/epoch_{}".format(epoch))
    torch.save(acc_clean, opt.acc_clean_dir + "/epoch_{}".format(epoch))
    torch.save(acc_bd, opt.acc_bd_dir + "/epoch_{}".format(epoch))

    # for tensorboard
    if not epoch % 1:
        tf_writer.add_scalars(
            "Clean Accuracy", {"Clean": avg_acc_clean, "Bd": avg_acc_bd, "Cross": avg_acc_cross}, epoch
        )
        tf_writer.add_image("Images", grid, global_step=epoch)

    schedulerC.step()


def eval(
    netC,
    optimizerC,
    schedulerC,
    test_dl,
    noise_grid,
    identity_grid,
    best_clean_acc,
    best_bd_acc,
    best_cross_acc,
    tf_writer,
    epoch,
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

    for batch_idx, (inputs, targets) in enumerate(test_dl):#一定在效果最好的时候保存
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
                targets_bd = torch.remainder(targets+1, opt.num_classes)#就是将所有的分类都往后推一个，都加一
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

                info_string = (
                    "Clean Acc: {:.4f} - Best: {:.4f} | Bd Acc: {:.4f} - Best: {:.4f} | Cross: {:.4f}".format(
                        acc_clean, best_clean_acc, acc_bd, best_bd_acc, acc_cross, best_cross_acc
                    )
                )
            else:
                info_string = "Clean Acc: {:.4f} - Best: {:.4f} | Bd Acc: {:.4f} - Best: {:.4f}".format(
                    acc_clean, best_clean_acc, acc_bd, best_bd_acc
                )
    print(info_string)
            # progress_bar(batch_idx, len(test_dl), info_string)

    # tensorboard
    if not epoch % 1:
        tf_writer.add_scalars("Test Accuracy", {"Clean": acc_clean, "Bd": acc_bd}, epoch)

    # Save checkpoint
    if acc_clean > best_clean_acc or (acc_clean > best_clean_acc - 0.1 and acc_bd > best_bd_acc):
        print(" Saving...")
        best_clean_acc = acc_clean
        best_bd_acc = acc_bd
        if opt.cross_ratio:
            best_cross_acc = acc_cross
        else:
            best_cross_acc = torch.tensor([0])
        state_dict = {
            "netC": netC.state_dict(),
            "schedulerC": schedulerC.state_dict(),
            "optimizerC": optimizerC.state_dict(),
            "best_clean_acc": best_clean_acc,
            "best_bd_acc": best_bd_acc,
            "best_cross_acc": best_cross_acc,
            "epoch_current": epoch,
            "identity_grid": identity_grid,
            "noise_grid": noise_grid,
        }
        torch.save(state_dict, opt.ckpt_path)
        with open(os.path.join(opt.ckpt_folder, "results.txt"), "w+") as f:
            results_dict = {
                "clean_acc": best_clean_acc.item(),
                "bd_acc": best_bd_acc.item(),
                "cross_acc": best_cross_acc.item(),
            }
            json.dump(results_dict, f, indent=2)

    return best_clean_acc, best_bd_acc, best_cross_acc

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

def main():
    # opt = config.get_arguments().parse_args()
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
    train_dl = get_dataloader(opt, True)
    test_dl = get_dataloader(opt, False)

    # prepare model
    netC, optimizerC, schedulerC = get_model(opt)

    # Load pretrained model
    mode = opt.attack_mode
    opt.ckpt_folder = os.path.join(opt.checkpoints, opt.dataset)#你选择的是哪种训练方式all2all或者all2one
    opt.ckpt_path = os.path.join(opt.ckpt_folder, "{}_{}_morph.pth.tar".format(opt.dataset, mode))
    opt.log_dir = os.path.join(opt.ckpt_folder, "log_dir")
    opt.loss_dir = os.path.join(opt.ckpt_folder, "Loss")
    opt.acc_clean_dir = os.path.join(opt.ckpt_folder, "Acc_clean")
    opt.acc_bd_dir = os.path.join(opt.ckpt_folder, "Acc_bd")
    if not os.path.exists(opt.log_dir):#此文件是否存在
        os.makedirs(opt.log_dir)#不存在，就创建路径

    if not os.path.exists(opt.loss_dir):
        os.makedirs(opt.loss_dir)#创建存储Loss的路径

    if not os.path.exists(opt.acc_clean_dir):
        os.makedirs(opt.acc_clean_dir)#创建存储acc_clean的路径

    if not os.path.exists(opt.acc_bd_dir):
        os.makedirs(opt.acc_bd_dir)#创建存储acc_bd的路径

    if opt.continue_training:
        if os.path.exists(opt.ckpt_path):
            print("Continue training!!")
            state_dict = torch.load(opt.ckpt_path)
            netC.load_state_dict(state_dict["netC"])
            optimizerC.load_state_dict(state_dict["optimizerC"])
            schedulerC.load_state_dict(state_dict["schedulerC"])
            best_clean_acc = state_dict["best_clean_acc"]
            best_bd_acc = state_dict["best_bd_acc"]
            best_cross_acc = state_dict["best_cross_acc"]
            epoch_current = state_dict["epoch_current"]
            identity_grid = state_dict["identity_grid"]
            noise_grid = state_dict["noise_grid"]
            tf_writer = SummaryWriter(log_dir=opt.log_dir)
        else:
            print("Pretrained model doesnt exist")
            exit()
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
        os.makedirs(opt.loss_dir)#创建存储Loss的路径
        os.makedirs(opt.acc_clean_dir)#创建存储acc_clean的路径
        os.makedirs(opt.acc_bd_dir)#创建存储acc_bd的路径
        with open(os.path.join(opt.ckpt_folder, "opt.json"), "w+") as f:
            json.dump(opt.__dict__, f, indent=2)
        tf_writer = SummaryWriter(log_dir=opt.log_dir)

    for epoch in range(epoch_current, opt.n_iters):
        print("Epoch {}:".format(epoch + 1))
        train(netC, optimizerC, schedulerC, train_dl, noise_grid, identity_grid, tf_writer, epoch, opt)
        best_clean_acc, best_bd_acc, best_cross_acc = eval(
            netC,
            optimizerC,
            schedulerC,
            test_dl,
            noise_grid,
            identity_grid,
            best_clean_acc,
            best_bd_acc,
            best_cross_acc,
            tf_writer,
            epoch,
            opt,
        )


if __name__ == "__main__":
    main()
