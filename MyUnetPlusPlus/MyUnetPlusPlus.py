import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import os
import segmentation_models_pytorch as smp
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' 

class seg_qyl(nn.Module):
    def __init__(self, model_name, n_class, encoder_weights=None):
        super().__init__()  
        self.model = smp.UnetPlusPlus(# UnetPlusPlus 
                encoder_name=model_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                
                encoder_weights=None,     # use `imagenet` pretrained weights for encoder initialization
                in_channels=3,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
                classes=n_class,                      # model output channels (number of classes in your dataset)
            )
    @autocast()
    def forward(self, x):
        #with autocast():
        x = self.model(x)
        return x

def load_unetP(n_class=11, model_name="efficientnet-b7", encoder_weights='imagenet'):
    model=seg_qyl(model_name,n_class,encoder_weights)
    return model

if __name__=="__main__":
    model=load_unetP()
    out=model(torch.zeros(5,3,128,128))
    print(out.size())