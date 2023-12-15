import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
import csv
from statistics import mean, stdev


def plot_models(history, path):

    y = history["train_loss"]
    y_1 = history["val_loss"]
    x = history["Epochs"].str.replace('Epoch ', '')
    plt.plot(x, y, label='Training')
    plt.plot(x, y_1, label='Validation')

    # Add in a title and axes labels
    plt.title('Training and Validation Loss')
    plt.xticks(np.arange(-1,60,10))
    plt.xlabel('Epochs', labelpad=10)
    plt.ylabel('Loss')
    # plt.axvline(x=best_model_epochs, color='g', linestyle=(0, (5, 1)),  label='best model')

    # Display the plot
    plt.legend(loc='best')
    plt.savefig(path + "loss_fig.png")

    plt.clf()

    y = history["train_acc"]
    y_1 = history["val_acc"]

    plt.plot(x, signal.savgol_filter(y, 60, 30), label='Training')
    plt.plot(x, signal.savgol_filter(y_1, 60, 30), label='Validation')

    # Add in a title and axes labels
    plt.title('Training and Validation Accuracy')
    plt.xticks(np.arange(-1, 60, 10))
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    # plt.axvline(x=best_model_epochs, color='g', linestyle=(0, (5, 1)), label='best model')

    # Display the plot
    plt.legend(loc='best')
    plt.savefig(path + "acc_fig.png")

    plt.clf()

    # plt.plot(history["iou"], label='Training')
    # plt.plot(history["val_iou"], label='Validation')
    #
    # # Add in a title and axes labels
    # plt.title('Training and Validation IOU')
    # plt.xlabel('Epochs')
    # plt.ylabel('IOU')
    # # plt.axvline(x=best_model_epochs, color='g', linestyle=(0, (5, 1)), label='best model')
    #
    # # Display the plot
    # plt.legend(loc='best')
    # plt.savefig(path + "iou_fig.png")
    #
    # plt.clf()


def joint_plot_models(history1, history2, path, best_model_epochs_1, best_model_epochs_2):

    plt.plot(history1["val_dice_coef"], label='UNet')
    plt.plot(history2["val_dice_coef"], label='D-UNet')

    # Add in a title and axes labels
    plt.title('Validation DICE')
    plt.xlabel('Epochs')
    plt.ylabel('Dice')
    plt.axvline(x=best_model_epochs_1, color='g', linestyle=(0, (5, 1)), label='best unet model')
    plt.axvline(x=best_model_epochs_2, color='purple', linestyle=(0, (5, 1)), label='best d_unet model')

    # Display the plot
    plt.legend(loc='best')
    plt.savefig(path + "dice_fig_vs.png")

    plt.clf()

    plt.plot(history1["val_iou"], label='UNet')
    plt.plot(history2["val_iou"], label='D-UNet')

    # Add in a title and axes labels
    plt.title('Validation IOU')
    plt.xlabel('Epochs')
    plt.ylabel('IOU')
    plt.axvline(x=best_model_epochs_1, color='g', linestyle=(0, (5, 1)), label='best unet model')
    plt.axvline(x=best_model_epochs_2, color='purple', linestyle=(0, (5, 1)), label='best d_unet model')

    # Display the plot
    plt.legend(loc='best')
    plt.savefig(path + "iou_fig_vs.png")

    plt.clf()


def compress(path, new_path, only_check_path):
    DF = []
    for cv in os.listdir(path):
        DF.append(pd.read_csv(path + cv))
    cont = 1
    f = open(new_path, 'w')
    f1 = open(only_check_path, 'w')
    epochs = 1000
    columns = ["Epochs", "loss", "std_loss", "iou", "std_iou", "dice_coef", "std_dice",
               "val_loss", "std_val_loss", "val_iou", "std_val_iou", "val_dice_coef", "std_val_dice"]
    writer = csv.writer(f)
    writer.writerow(columns)
    writer1 = csv.writer(f1)
    writer1.writerow(columns)
    for i in range(epochs):
        I = []
        D = []
        L = []
        vI = []
        vD = []
        vL = []
        for d in DF:
            L.append(d.iloc[i]['loss'])
            I.append(d.iloc[i]['iou'])
            D.append(d.iloc[i]['dice_coef'])
            vL.append(d.iloc[i]['val_loss'])
            vI.append(d.iloc[i]['val_iou'])
            vD.append(d.iloc[i]['val_dice_coef'])
        avgL = mean(L)
        avgI = mean(I)
        avgD = mean(D)
        avgvL = mean(vL)
        avgvI = mean(vI)
        avgvD = mean(vD)

        sL = stdev(L)
        sI = stdev(I)
        sD = stdev(D)
        svL = stdev(vL)
        svI = stdev(vI)
        svD = stdev(vD)

        r = [str(cont), str(avgL), str(sL), str(avgI), str(sI), str(avgD), str(sD),
             str(avgvL), str(svL), str(avgvI), str(svI), str(avgvD), str(svD)]
        writer.writerow(r)
        if cont % 100 == 0:
            writer1.writerow(r)
        cont += 1
    f.close()
    f1.close()


def get_best(m_path):
    df = pd.read_csv(m_path)
    return int(df.iloc[df.idxmax()["val_dice_coef"], :]["Epochs"])


if __name__ == "__main__":

    results_path = Path('results')
    vit_path = results_path / 'ViT'
    cnn_normal_path = results_path / 'CNN'
    ode_path = results_path / 'NeuralODECNNClassifier'
    if not results_path.exists():
        results_path.mkdir()
    if not vit_path.exists():
        vit_path.mkdir()
    if not cnn_normal_path.exists():
        cnn_normal_path.mkdir()
    if not ode_path.exists():
        ode_path.mkdir()

    path = 'results/NeuralODECNNClassifier/'
    visual_path = 'results/unet_normal_visual/'
    new_path_1 = 'backbone_resume_root/NeuralODECNNClassifier/resume.csv'
    only_check_path = 'results/unet_normal_visual/mean_history_checkpoints.csv'

    # compress(path, new_path_1, only_check_path)
    # bst = get_best(only_check_path)
    # print(bst)
    plot_models(pd.read_csv(new_path_1), path)

    # path = 'results/unet_32/'
    # visual_path = 'results/unet_32_visual/'
    # new_path_2 = 'results/unet_32_visual/mean_history.csv'
    # new_write = 'results/unet_32_visual/table.csv'
    # only_check_path = 'results/unet_32_visual/mean_history_checkpoints.csv'
    #
    # compress(path, new_path_2, only_check_path)
    # bst = get_best(only_check_path)
    # print(bst)
    # plot_models(pd.read_csv(new_path_2), visual_path, best_model_epochs=bst)
    #
    # joint_plot_models(pd.read_csv(new_path_1), pd.read_csv(new_path_2), path='results/',
    #                   best_model_epochs_1=bst, best_model_epochs_2=bst)

