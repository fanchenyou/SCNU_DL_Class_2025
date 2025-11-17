import torch
import torch.nn.functional as F
import cv2
import torchvision.transforms as transforms
from torchvision.utils import save_image



###############################################
## Warning: Do not modify any lines below  ###
###############################################
# Read the image
image = cv2.imread('porche.png')
# Convert BGR image to RGB image
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Define a transform to convert the image to torch tensor
transform = transforms.Compose([transforms.ToTensor()])
# Convert the image to Torch tensor, and make it [B=1, C=3, H, W]
I = transform(image)
I = I.unsqueeze(0)
print("原始图像尺寸:",I.size())
###############################################




###############################################
## TODO: move the car to 200 pixels up and left
###############################################
# Design a 2D kernel H_2d to move the car to 200 pixels up and left
# H_2d = torch.zeros(200,200).float()  # modify this
# TODO: assign H_2d values and perform 2-D conv as code below (modify them if necessary)
# https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html


# Design two 1-D kernels H_1d_a and H_1d_b to move the car to 200 pixels up first, then left, in separate fashion
# H_1d_horizon = torch.zeros(1,200).float()  # modify this
# H_1d_vertical = torch.zeros(200,100).float()  # modify this
# TODO: assign kernel values and perform 1-D conv as code below (modify them if necessary)
'''
out2 = F.conv2d(I, H_1d_horizon, groups=?, stride=[1,1], padding=[?,?])
print(out1.size())
# apply transform on vertical axis
out3 = F.conv2d(out2, H_1d_vertical, groups=?, stride=[1,1], padding=[?,?])
print(out2.size())
save_image(out3, 'porche_move_two_1d.png')
'''


###############################################
## DONE: 1D-start
###############################################
kernel_size_vertical = 201
H_1d_vertica = torch.zeros(3, 1, kernel_size_vertical, 1).float() 
H_1d_vertica[:, 0, 200, 0] = 1.0  

padding_vertical = (kernel_size_vertical - 1) // 2  
out2 = F.conv2d(I, H_1d_vertica, groups=3, padding=[padding_vertical, 0])
print("垂直1D卷积后尺寸:", out2.size())
save_image(out2, 'porche_move_two_1d_1.png')

kernel_size_horizontal = 201
H_1d_horizon = torch.zeros(3, 1, 1, kernel_size_horizontal).float()
H_1d_horizon[:, 0, 0,200] = 1.0  

padding_horizontal = (kernel_size_horizontal - 1) // 2  
out3 = F.conv2d(out2, H_1d_horizon, groups=3, padding=[0, padding_horizontal])
print("水平1D卷积后尺寸:", out3.size())
save_image(out3, 'porche_move_two_1d_2.png')
###############################################
## DONE: 1D-end
###############################################