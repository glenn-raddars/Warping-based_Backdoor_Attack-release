import torch
import matplotlib.pyplot as plt

def plot_loss(n):
    y = []
    
    for i in range(n):
        enc = torch.load("./all2all_all_Blending/mnist/Loss/epoch_{}".format(i))
        tempy = enc.item()
        #print(tempy)
        y.append(tempy)
        
    x = range(0, len(y))
    plt.plot(x, y, '.-')
    plt.title("the loss")
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.savefig('./all2all_all_Blending/mnist/lossImg')
    plt.show()

def plot_accuracy(n):
    y_acc_bd = []
    y_acc_clean = []

    for i in range(n):
        enc_clean = torch.load("./all2all_all_Blending/mnist/Acc_clean/epoch_{}".format(i))
        enc_bd = torch.load("./all2all_all_Blending/mnist/Acc_bd/epoch_{}".format(i))

        y_acc_clean.append(enc_clean.item())
        y_acc_bd.append(enc_bd.item())

    x = range(0, len(y_acc_bd))
    plt.plot(x, y_acc_clean, 'r.-')
    plt.plot(x, y_acc_bd, 'b.-')
    plt.title('the Accurancy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig('./all2all_all_Blending/mnist/AccImg')
    plt.show()

if __name__ == "__main__":
    plot_loss(600)
    plot_accuracy(600)
