import torch
import torch.nn.functional as F
import cv2
import torchvision.transforms as transforms
from torchvision.utils import save_image



###############################################
## Warning: Do not modify any lines below  ###
###############################################
# Read the image
image = cv2.imread('porche.jpg')
# Convert BGR image to RGB image
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Define a transform to convert the image to torch tensor
transform = transforms.Compose([transforms.ToTensor()])
# Convert the image to Torch tensor, and make it [B=1, C=3, H, W]
I = transform(image)
I = I.unsqueeze(0)
print(I.size())
###############################################




###############################################
## TODO: move the car to 200 pixels up and left
###############################################

# Design a 2D kernel H_2d to move the car to 200 pixels up and left
H_2d = torch.zeros(200,200).float()  # modify this
# TODO: assign H_2d values and perform 2-D conv as code below (modify them if necessary)
# https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html
'''
# out1 = F.conv2d(I, H_2d, padding=[?,?])
# save_image(out1, 'porche_move_2d.png')
'''

# Design two 1-D kernels H_1d_a and H_1d_b to move the car to 200 pixels up first, then left, in separate fashion
H_1d_horizon = torch.zeros(1,200).float()  # modify this
H_1d_vertical = torch.zeros(200,100).float()  # modify this
# TODO: assign kernel values and perform 1-D conv as code below (modify them if necessary)
'''
out2 = F.conv2d(I, H_1d_horizon, groups=?, stride=[1,1], padding=[?,?])
print(out1.size())
# apply transform on vertical axis
out3 = F.conv2d(out2, H_1d_vertical, groups=?, stride=[1,1], padding=[?,?])
print(out2.size())
save_image(out3, 'porche_move_two_1d.png')
'''
