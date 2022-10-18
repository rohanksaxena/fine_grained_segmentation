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
from PIL import Image
# from model.model import SSNModel
import cv2


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
    pred = custom_forward(model, img)
    # print(f'Prediction shape: {pred.shape}')
    # # pred = torch.reshape(pred, (64, 56, 56))
    # img = pred.squeeze(0).cpu().numpy()
    # maps = img.reshape((img.shape[0]), -1).T
    # print(f'Map shape: {maps.shape}')
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # K = 5
    # attempts = 10
    # ret, label, center = cv2.kmeans(maps, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    # result_image = label.reshape((img.shape[1], img.shape[2]))
    # print(f'Map shape: {result_image.shape}')
    # plt.imshow(result_image)
    # plt.axis('off')
    # plt.savefig(f'clustered.jpg')
    return pred


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
        print(model)
        print(f'Device: {device}')
        extracted_features = predict(model, device)
    print(f'Feature Extraction complete, feature shape: {extracted_features.shape}')

    # # Train SSN
    # model = SSNModel(n_spix=100, training=True)
    # print(model.parameters())
    # optimizer = optim.Adam(model.parameters(), lr=1e-4)
