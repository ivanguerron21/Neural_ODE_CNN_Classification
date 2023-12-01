from pathlib import Path

import numpy as np
import torch
import os
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

from NeuralODE import NeuralODECNNClassifier, NeuralODE, ConvODEF
from utils import resume_macro
import torchvision.datasets as datasets

os.system('color')
HEAD_NAME = "CosFace"  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SFaceLoss']

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GPU_ID = None
test_dir = "split_data/test"

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def pre_image(image_path, model, transform_norm):
    labels = torch.tensor(range(32))
    labels = labels.to(DEVICE).long()
    img = Image.open(image_path)
    img_normalized = transform_norm(img).float()
    img_normalized = img_normalized.unsqueeze_(0)
    img_normalized = img_normalized.to(DEVICE)
    with torch.no_grad():
        model.eval()
        output = model(img_normalized, labels)
        _, index = torch.max(output.data, 1)
        classes = test_dataset.classes
        class_name = classes[index]
        return class_name


def denormal(image):
    image = image.numpy().transpose(1, 2, 0)

    image = image * std + mean
    image = np.clip(image, 0, 1)
    return image


def denormalize(x):
    # 3, H, W, B
    ten = x.clone().permute(1, 2, 3, 0)
    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)
    # B, 3, H, W
    return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)


data_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(test_dir, transform=data_transform)
conv_dim = 8
NUM_CLASS = len([i for i in os.listdir(test_dir) if not i.startswith('.')])
model = NeuralODECNNClassifier(NeuralODE(ConvODEF(conv_dim * 4)),
                                  out_dim=NUM_CLASS, conv_dim=conv_dim, loss_type=HEAD_NAME, device=DEVICE)

classes = test_dataset.classes
model.load_state_dict(torch.load('checkpoints/Backbone_NeuralODECNNClassifier_Epoch_5_Time_2023-11-10-18-24_checkpoint.pth'))
model.eval()
test_loader = DataLoader(dataset=test_dataset, batch_size=32,
                         shuffle=True, num_workers=0, drop_last=True)

pred = {}
vP = 0
vR = 0
vF = 0
vA = 0
if not Path('results_test').exists():
    Path('results_test').mkdir()
for i, (inputs, labels) in enumerate(iter(test_loader)):
    inputs = inputs.to(DEVICE)
    labels = labels.to(DEVICE).long()
    outputs, emb = model(inputs.float(), labels)
    _, predicted = torch.max(outputs.data, 1)
    if i == 0:
        pred = predicted[0]
        class_pred = []
        denorm_images = denormalize(inputs.cpu())
        for i in range(len(predicted)):
            class_pred = classes[predicted[i]]
            fig = plt.figure(figsize=(4, 4))
            img = denorm_images[i].permute(1, 2, 0).clamp(0, 1)
            fig.suptitle(class_pred.replace("pins_", "").upper())
            plt.imshow(img)
            plt.savefig(f'results_test/{i}')
            plt.clf()
    P_t, R_t, F_t, A_t = resume_macro(labels.cpu(), predicted.cpu())
    vP += P_t
    vR += R_t
    vF += F_t
    vA += A_t

total_accuracy = vA / len(test_loader)

print("Test accuracy: ", total_accuracy)