import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.utils import load_state_dict_from_url


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):

        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        return outputs

cfgs = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
}
def make_layers(cfg, batch_norm=False, in_channels = 3):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
    
class Unet(nn.Module):
    def __init__(self, num_classes=11, in_channels=3, pretrained=False,in_filters = [192, 384, 768, 1024],out_filters = [64, 128, 256, 512]):
        super(Unet, self).__init__()
        # self.vgg = model = VGG(pretrained=pretrained,in_channels=in_channels)
        self.features=make_layers(cfgs['D'],)
        # upsampling
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        # final conv (without any concat)
        self.final = nn.Conv2d(out_filters[0], num_classes, 1)
        
    def forward(self, inputs):
        feat1 = self.features[  :4 ](inputs)
        feat2 = self.features[4 :9 ](feat1)
        feat3 = self.features[9 :16](feat2)
        feat4 = self.features[16:23](feat3)
        feat5 = self.features[23:-1](feat4)

        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        final = self.final(up1)
        
        return final

    # def _initialize_weights(self, *stages):
    #     for modules in stages:
    #         for module in modules.modules():
    #             if isinstance(module, nn.Conv2d):
    #                 nn.init.kaiming_normal_(module.weight)
    #                 if module.bias is not None:
    #                     module.bias.data.zero_()
    #             elif isinstance(module, nn.BatchNorm2d):
    #                 module.weight.data.fill_(1)
    #                 module.bias.data.zero_()

if __name__=="__main__":
    unet=Unet()#
    # print(unet)
    ot=unet(torch.zeros(5,3,256,256))
    print(ot.shape)