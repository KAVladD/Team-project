import os
import time

import torch, torchvision
from matplotlib import pyplot as plt

from get_points import get_points
from draw_signal import draw_signal
from utils import *

task = 1
output = "norm_dist" #dist, norm_dist or angle
height = 1280
width = 720

#video_path = "test_vids/IMG_8469.MOV"
video_path = "test_vids/IMG_8471.MOV"

weights_path = "ep_5_with_norm.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"

net1 = torchvision.models.resnet18(pretrained=True)
net1.fc = torch.nn.Linear(512, 1)
net1 = net1.to(device)
net1.load_state_dict(torch.load(weights_path))
net1.eval()

val_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(size=(224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

cap = cv2.VideoCapture(video_path)
net1_signal = []
net2_signal = []
n = 0

while True:

    ret, frame = cap.read()
    if not ret:
        break

    n += 1

    image = Image.fromarray(frame)
    image = val_transform(image).unsqueeze(0).to(device)

    res1 = net1(image)
    net1_signal.append(res1.to("cpu").detach().numpy()[0][0])

plt.plot(net1_signal)
plt.ylim((0.1, 0.6))
plt.title("Зависимость выхода сети от номера кадра (изменение положения руки)")
plt.show()