import csv
import os
from pathlib import Path
from NeuralODE import NeuralODECNNClassifier, NeuralODE, ConvODEF
from utils import get_time, resume_macro
import torch
import torch.nn as nn
import pandas as pd
from types import SimpleNamespace
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    SEED = 37  # random seed for reproduce results
    torch.manual_seed(SEED)
    args = SimpleNamespace()
    args.weight_decay = 0.05
    args.opt = 'adamw'  # 'lookahead_adam' to use `lookahead`
    args.momentum = 0.9
    args.epochs = 100
    args.lr = 5e-4
    args.lr_noise = None
    args.lr_noise_pct = 0.67
    args.lr_noise_std = 1.0
    args.warmup_lr = 1e-6
    args.min_lr = 1e-5
    args.decay_epochs = 30
    args.warmup_epochs = 3
    args.cooldown_epochs = 10
    args.patience_epochs = 10
    args.decay_rate = 0.1
    args.sched = "cosine"

    WORK_PATH = "checkpoints"
    checkpoint = ''
    BACKBONE_RESUME_ROOT = "backbone_resume_root"
    PRE_TRAINED = False
    results_path = "results"
    BACKBONE_NAME = "NeuralODECNNClassifier"
    HEAD_NAME = 'CosFace'
    IMAGE_SIZE = 32

    if not Path(WORK_PATH).exists():
        Path(WORK_PATH).mkdir()
    if not Path(results_path).exists():
        Path(results_path).mkdir()
    if not Path(results_path + "/" + BACKBONE_NAME).exists():
        Path(results_path + "/" + BACKBONE_NAME).mkdir()
    if not Path(BACKBONE_RESUME_ROOT).exists():
        Path(BACKBONE_RESUME_ROOT).mkdir()
    if not Path(BACKBONE_RESUME_ROOT + "/" + BACKBONE_NAME).exists():
        Path(BACKBONE_RESUME_ROOT + "/" + BACKBONE_NAME).mkdir()

    BATCH_SIZE = 32
    NUM_EPOCH = 60
    conv_dim = 8

    # DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("DEVICE", DEVICE)

    torch.backends.cudnn.benchmark = True
    train_dir = "split_data/train"
    val_dir = "split_data/val"

    NUM_CLASS = len([i for i in os.listdir(train_dir) if not i.startswith('.')])

    num_workers = 0
    data_transform = transforms.Compose([
        transforms.Resize(size=(IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=data_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=data_transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=num_workers, drop_last=True)

    print("Number of Training Classes: {}".format(NUM_CLASS))

    BACKBONE = NeuralODECNNClassifier(NeuralODE(ConvODEF(conv_dim*4)),
                                      out_dim=NUM_CLASS, conv_dim=conv_dim, loss_type=HEAD_NAME, device=DEVICE)
    LOSS = nn.CrossEntropyLoss()

    OPTIMIZER = create_optimizer(args, BACKBONE)
    lr_scheduler, _ = create_scheduler(args, OPTIMIZER)

    BACKBONE.to(DEVICE)
    if PRE_TRAINED:
        BACKBONE.load_state_dict(torch.load(checkpoint))
        n = int(checkpoint.split("/")[1].split('_')[3])
        df = pd.read_csv(f'{BACKBONE_RESUME_ROOT}/{BACKBONE_NAME}' + '/resume.csv')
        df.set_index("Epochs", inplace=True)
        val_best = df.loc[f'Epoch {n}']['val_acc']
    else:
        f = open(f'{BACKBONE_RESUME_ROOT}/{BACKBONE_NAME}' + '/resume.csv', 'w')
        columns = ["Epochs", "train_loss", "val_loss", "train_acc", "val_acc", "train_pre", "val_pre", "train_rec",
                   "val_rec", "train_f1", "val_f1"]
        writer = csv.writer(f)
        writer.writerow(columns)
        f.close()
        val_best = 0.5
        n = 0
    step_print = 8
    BACKBONE.train()  # set to training mode
    for epoch in range(n, NUM_EPOCH):  # start training process
        f = open(f'{BACKBONE_RESUME_ROOT}/{BACKBONE_NAME}' + '/resume.csv', 'a')
        writer = csv.writer(f)
        batch = 0.0
        lr_scheduler.step(epoch)
        P, R, F, A = 0.0, 0.0, 0.0, 0.0
        vP, vR, vF, vA = 0.0, 0.0, 0.0, 0.0
        running_train_loss = 0.0
        running_val_loss = 0.0
        last_time = time.time()

        for i, (inputs, labels) in enumerate(iter(train_loader)):
            # compute output
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE).long()
            if HEAD_NAME is not None:
                outputs, emb = BACKBONE(inputs.float(), labels)
            else:
                outputs = BACKBONE(inputs.float())
            loss = LOSS(outputs, labels)
            OPTIMIZER.zero_grad()
            loss.backward()
            OPTIMIZER.step()

            _, predicted = torch.max(outputs.data, 1)
            P_t, R_t, F_t, A_t = resume_macro(labels.cpu(), predicted.cpu())
            P += P_t
            R += R_t
            F += F_t
            A += A_t
            running_train_loss += loss.item()
            if (i + 1) % 8 == 0:
                batch_time = time.time()
                print('Epoch {} Batch {}\t'
                      'Time: {time:.2f} s\t'
                      'Training Loss {loss:.4f}\t'
                      'Training Acc {top1:.3f}'
                      .format(epoch + 1, i + 1, time=batch_time - last_time,
                              loss=loss.item(), top1=A_t))

        train_loss_value = running_train_loss / len(train_loader)
        train_accuracy = A / len(train_loader)
        train_pre = (P / len(train_loader)).__round__(3)
        train_recall = (R / len(train_loader)).__round__(3)
        train_f1 = (F / len(train_loader)).__round__(3)

        with torch.no_grad():
            BACKBONE.eval()
            for img, lab in val_loader:
                inputs, labels = img.to(DEVICE), lab.to(DEVICE).long()
                if HEAD_NAME is not None:
                    predictions, emb = BACKBONE(inputs.float(), labels)
                else:
                    predictions = BACKBONE(inputs.float())
                val_loss = LOSS(predictions, labels)

                _, val_predicted = torch.max(predictions.data, 1)
                running_val_loss += val_loss.item()
                P_t, R_t, F_t, A_t = resume_macro(labels.cpu(), val_predicted.cpu())
                vP += P_t
                vR += R_t
                vF += F_t
                vA += A_t

        val_loss_value = running_val_loss / len(val_loader)
        val_accuracy = vA / len(val_loader)
        val_pre = (vP / len(val_loader)).__round__(3)
        val_recall = (vR / len(val_loader)).__round__(3)
        val_f1 = (vF / len(val_loader)).__round__(3)
        writer.writerow([f'Epoch {epoch + 1}', train_loss_value, val_loss_value, train_accuracy, val_accuracy,
                         train_pre, val_pre, train_recall, val_recall, train_f1, val_f1])
        epoch_time = time.time()
        print('Epoch {}\t'
              'Time: {time:.2f} s\t'
              'Training Loss {loss:.4f}\t'
              'Val Loss {v_loss:.4f}\t'
              'Training Acc {top1:.3f}\t'
              'Val Acc {topv:.3f}\t'
              .format(epoch + 1, time=epoch_time - last_time,
                      loss=train_loss_value, v_loss=val_loss_value, top1=train_accuracy, topv=val_accuracy))

        if val_accuracy > val_best or val_accuracy > 0.95:
            torch.save(BACKBONE.state_dict(),
                       os.path.join(WORK_PATH, "Backbone_{}_Epoch_{}_Time_{}_checkpoint.pth"
                                    .format(BACKBONE_NAME, epoch + 1, get_time())))

            val_best = val_accuracy
        BACKBONE.train()  # set to training mode
        f.close()
