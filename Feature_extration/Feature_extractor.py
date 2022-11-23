import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

class Transpose1dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=11, upsample=None, output_padding=1):
        super(Transpose1dLayer, self).__init__()
        self.upsample = upsample
        self.upsample_layer = torch.nn.Upsample(scale_factor=upsample)
        reflection_pad = kernel_size // 2
        self.reflection_pad = nn.ConstantPad1d(reflection_pad, value=0)
        self.conv1d = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.Conv1dTrans = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, output_padding)

    def forward(self, x):
        if self.upsample:
            #x = torch.cat((x, in_feature), 1)
            return self.conv1d(self.reflection_pad(self.upsample_layer(x)))
        else:
            return self.Conv1dTrans(x)


class Feature_extractor(nn.Module):
    def __init__(self,latent_dim=100, post_proc_filt_len=512,upsample=True):
        super(Feature_extractor, self).__init__()
        # "Dense" is the same meaning as fully connection.
        stride = 4
        if upsample:
            stride = 1
            upsample = 5
        # if upsample is anything but none Transpose1dLayer will do
        # self.conv1d(self.reflection_pad(self.upsample_layer(x)))
        # which is a 1d convolution on padded and upsampled data x
        self.deconv_1 = Transpose1dLayer(250 , 250, 25, stride, upsample=upsample)
        self.deconv_2 = Transpose1dLayer(250, 150, 25, stride, upsample=upsample)
        self.deconv_3 = Transpose1dLayer(150, 50, 25, stride, upsample=upsample)
        self.deconv_4 = Transpose1dLayer( 50, 25, 25, stride, upsample=2)
        self.deconv_5 = Transpose1dLayer( 25, 10, 25, stride, upsample=upsample)
        self.deconv_6 = Transpose1dLayer(  10, 7, 25, stride, upsample=2)


        #new convolutional layers
        self.conv_1 = nn.Conv1d(7, 10, 25, stride=2, padding=25 // 2)
        self.conv_2 = nn.Conv1d(10, 25, 25, stride=5, padding= 25 // 2)
        self.conv_3 = nn.Conv1d(25, 50 , 25, stride=2, padding= 25 // 2)
        self.conv_4 = nn.Conv1d(50, 150 , 25, stride=5, padding= 25 // 2)
        self.conv_5 = nn.Conv1d(150, 250 , 25, stride=5, padding= 25 // 2)
        self.conv_6 = nn.Conv1d(250, 250 , 25, stride=5, padding= 25 // 2)
        self.flatt = nn.Flatten()
        self.linear1 = nn.Linear(500,100)
        self.linear2 = nn.Linear(100,500)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x, LS=False):
        self.LS=LS
        if x.ndim==2:
          x=x.unsqueeze(0)
        x = F.leaky_relu(self.conv_1(x)) #(1,7,5000 --> 1, 10, 2500)
        x = F.leaky_relu(self.conv_2(x)) #( --> 1, 25, 500)
        x = F.leaky_relu(self.conv_3(x)) #(--> 1, 50, 250)
        x = F.leaky_relu(self.conv_4(x)) # --> 1, 150, 50)
        x = F.leaky_relu(self.conv_5(x)) #(--> 1, 250, 10)
        x = F.leaky_relu(self.conv_6(x)) #(--> 1, 250, 2)-->flatten into (1,500)), then to linear ((1,100)), and then back
        x = self.flatt(x) # (1,500)
        LS = self.linear1(x) #(1,100)
        if self.LS is True:
          return LS
        x = self.linear2(LS) #(1,500)
        zero_dim=x.shape[0]
        x=torch.reshape(x,(zero_dim,250,2)) #1(1,250,2)
        x = F.relu(self.deconv_1(x)) #(--> 1, 250, 10)
        x = F.relu(self.deconv_2(x)) #(--> 1, 150, 50)
        x = F.relu(self.deconv_3(x)) #( --> 1, 50, 250)
        x = F.relu(self.deconv_4(x)) #(--> 1, 25, 500)
        x = F.relu(self.deconv_5(x)) #(--> 1, 10, 2500)
        x = torch.tanh(self.deconv_6(x)) #(1, 7, 5000)
        x=x.squeeze()
        return x


