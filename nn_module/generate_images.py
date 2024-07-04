import os

import torch, torchvision
from matplotlib import pyplot as plt

from utils import *
from utils_nn import  init_train_transform, init_val_transform
from utils_nn import init_gray_train_transform, init_gray_val_transform

video_path = "new_vids/patient0/m1/1_R.MOV"

weights_path = "ep_5_with_norm.pth" #"weights/gray_ep_100_angle_new_data_lr_6/best.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"

net1 = torchvision.models.resnet18(pretrained=True)
net1.fc = torch.nn.Linear(512, 1)
#net1.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
net1 = net1.to(device)
net1.load_state_dict(torch.load(weights_path))
net1.eval()

net2 = torchvision.models.resnet18(pretrained=True)
net2.fc = torch.nn.Linear(512, 1)
#net2.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
net2 = net2.to(device)
net2.eval()

val_transform = init_val_transform()

signal_path, frames_path = pars_video_path(video_path)

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

    res2 = net2(image)
    net2_signal.append(res2.to("cpu").detach().numpy()[0][0])

gt_signal = np.load(signal_path, allow_pickle=True)
frames = np.load(frames_path, allow_pickle=True)

gt_impr_signal = []
k = 0

for i in range(n):
    if i in frames:
        gt_impr_signal.append(gt_signal[k])
        k += 1
    else:
        gt_impr_signal.append(0)

norm_gt_impr_signal = norm_signal(gt_impr_signal)
norm_net1_signal = norm_signal(net1_signal)
norm_net2_signal = norm_signal(net2_signal)

fig, ax = plt.subplots(nrows=2, ncols=1)
ax[0].plot(norm_gt_impr_signal, label="MediaPipe")
ax[0].plot(norm_net1_signal, label="Обученное решение")
ax[0].legend()
ax[1].plot(norm_gt_impr_signal, label="MediaPipe")
ax[1].plot(norm_net2_signal, label="Случайная инициализация")
ax[1].legend()
plt.show()

# plt.plot(norm_gt_impr_signal)
# plt.plot(norm_net1_signal)
# plt.plot(norm_net2_signal)
# plt.show()