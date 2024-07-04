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

video_path = "new_vids/patient0/m1/1_L.MOV"

weights_path = "weights/ep_20_angle_new_data_lr_5/best.pth"

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


start_time = time.time()
dots = get_points(video_path, grap=False)
signal, frames = draw_signal(dots, task=task, output=output, height=height, width=width)
end_time = time.time()

print(end_time - start_time)


start_time = time.time()
while True:

    ret, frame = cap.read()
    if not ret:
        break

    n += 1

    image = Image.fromarray(frame)
    image = val_transform(image).unsqueeze(0).to(device)

    res1 = net1(image)
    net1_signal.append(res1.to("cpu").detach().numpy()[0][0])
end_time = time.time()

print(end_time - start_time)
print(n)