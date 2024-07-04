import os

import torch, torchvision
from matplotlib import pyplot as plt

from utils import *

vid1_path = "new_vids/patient0/m1/1_L.MOV"
vid2_path = "new_vids/patient0/m1/1_R.MOV"
# vid_path = "fingers_vids/Patient83/m1/1_L.MOV"

weights_path = "ep_10_with_norm.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"

net1 = torchvision.models.resnet18(pretrained=True)
net1.fc = torch.nn.Linear(512, 1)
net1 = net1.to(device)
net1.load_state_dict(torch.load(weights_path))
net1.eval()

net2 = torchvision.models.resnet18(pretrained=True)
net2.fc = torch.nn.Linear(512, 1)
net2 = net2.to(device)
net2.eval()

val_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(size=(224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

res1 = np.array([process_video(vid1_path, net1, net2, val_transform)])
res2 = np.array([process_video(vid2_path, net1, net2, val_transform)])

print((res1 + res2) / 2 * 0.033)