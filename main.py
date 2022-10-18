import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
from PIL import Image
from model.model import SSNModel
import cv2
from lib.dataset import bsds, augmentation


def predict(model, device):
    input_size = (224, 224)
    img_path = os.path.join(os.getcwd(), 'Sooty_Albatross_0031_1066.jpg')
    val_transform = T.Compose([
        T.Resize(input_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    model.to(device)
    img = Image.open(img_path).convert('RGB')
    img = val_transform(img).unsqueeze(0)
    img = img.to(device)
    print(f'Image shape: {img.shape}')
    features = custom_forward(model, img)
    return features


def custom_forward(model, x):
    # See note [TorchScript super()]
    input_size = x.shape[-1]
    input = x
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    l1 = model.layer1(x)
    l2 = model.layer2(l1)
    l3 = model.layer3(l2)
    l4 = model.layer4(l3)

    l1 = nn.functional.interpolate(l1, size=input_size, mode='bilinear', align_corners=False)
    l2 = nn.functional.interpolate(l2, size=input_size, mode='bilinear', align_corners=False)
    l3 = nn.functional.interpolate(l3, size=input_size, mode='bilinear', align_corners=False)
    l4 = nn.functional.interpolate(l4, size=input_size, mode='bilinear', align_corners=False)

    cat_feat = torch.cat([input, l1, l2, l3, l4], 1)

    # x = model.avgpool(x)
    # x = torch.flatten(x, 1)
    # x = model.fc(x)

    return cat_feat


if __name__ == '__main__':
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    with torch.no_grad():
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Device: {device}')
        extracted_features = predict(model, device)
    print(f'Feature Extraction complete, feature shape: {extracted_features.shape}')

    # Load Data
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    augment = augmentation.Compose(
        [augmentation.RandomHorizontalFlip(), augmentation.RandomScale(), augmentation.RandomCrop()])
    train_dataset = bsds.BSDS('/mnt/nfs-students/fine_grained_segmentation/BSDS500/BSDS500', geo_transforms=augment)
    train_loader = DataLoader(train_dataset, 6, shuffle=True, drop_last=True, num_workers=4)

    test_dataset = bsds.BSDS('/mnt/nfs-students/fine_grained_segmentation/BSDS500/BSDS500', split="val")
    test_loader = DataLoader(test_dataset, 1, shuffle=False, drop_last=False)

    # Train model
    model = SSNModel(n_spix=100, training=True)
    print(model.parameters())
