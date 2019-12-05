from model import * 
from torchsummary import summary

LR = 0.001

DEVICE = torch.device("cuda")

BASE ="check"
INP_SIZE = 1280
IMAGE_SIZE = 224
    
model, optimizer, loss = prepare_eff_model(lr=LR, device=DEVICE, name="efficientnet_b1", inp_size = INP_SIZE, im_size=IMAGE_SIZE)
summary(model, input_size = (6, 224, 224))