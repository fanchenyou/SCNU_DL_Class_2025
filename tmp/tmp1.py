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
print(I.size())
###############################################




###############################################



# Design two 1-D kernels H_1d_a and H_1d_b to move the car to 200 pixels up first, then left, in separate fashion
H_1d_horizon = torch.zeros(1,201).float()  # modify this
H_1d_vertical = torch.zeros(201,101).float()  # modify this
# TODO: assign kernel values and perform 1-D conv as code below (modify them if necessary)
H_1d_horizon[0][200]=1
H_1d_vertical[200][50]=1
H_1d_horizon=torch.stack((H_1d_horizon,H_1d_horizon,H_1d_horizon),0)
H_1d_horizon=torch.reshape(H_1d_horizon,[3,1,H_1d_horizon.size()[1],H_1d_horizon.size()[2]])
H_1d_vertical=torch.stack((H_1d_vertical,H_1d_vertical,H_1d_vertical),0)
H_1d_vertical=torch.reshape(H_1d_vertical,[3,1,H_1d_vertical.size()[1],H_1d_vertical.size()[2]])

out2 = F.conv2d(I, H_1d_horizon, groups=3, stride=[1,1], padding=[0,100])
# apply transform on vertical axis
out3 = F.conv2d(out2, H_1d_vertical, groups=3, stride=[1,1], padding=[100,50])
print(out2.size())
save_image(out3, 'porche_move_two_1d.png')

