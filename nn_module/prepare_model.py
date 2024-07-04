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

model = torch.jit.script(net1)
model._save_for_lite_interpreter(weights_path.split("/")[1] + "_mobile_model.ptl")