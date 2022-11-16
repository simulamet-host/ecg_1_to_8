 # Modified version of puls2puls to accomodate for Diffusion model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class up_first_last(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=11, upsample=None, output_padding=1):
        super(up_first_last, self).__init__()
        self.upsample = upsample
        self.upsample_layer = torch.nn.Upsample(scale_factor=upsample)
        reflection_pad = kernel_size // 2
        self.reflection_pad = nn.ConstantPad1d(reflection_pad, value=0)
        self.conv1d = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.Conv1dTrans = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, output_padding)

    def forward(self, x):
        if self.upsample:
          print(f"x_input_{x.shape}")
          x = self.conv1d(self.reflection_pad(self.upsample_layer(x)))
          print(f"x_output_{x.shape}")
          return x
        else:
          return self.Conv1dTrans(x)

class up(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=11, upsample=None, output_padding=1,emb_dim=256):
        super(up, self).__init__()
        self.upsample = upsample

        self.upsample_layer = torch.nn.Upsample(scale_factor=upsample)
        reflection_pad = kernel_size // 2
        self.reflection_pad = nn.ConstantPad1d(reflection_pad, value=0)
        self.conv1d = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.Conv1dTrans = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, output_padding)

#         # this is for t to be transformed 
#         self.emb_layer = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(
#                 emb_dim,  ## this should be the inital size of t
#                 out_channels
# #            ),
# #        )

    def forward(self, x, x_skip,t,y):
        print("we are in up")
        if self.upsample:
            x = torch.cat((x, x_skip), 1)
            x=self.conv1d(self.reflection_pad(self.upsample_layer(x)))
            emb = (t)[:, :, None].repeat(1, x.shape[-2], x.shape[-1])
            if y is not None:
              reshape_layer=nn.Linear(y.shape[-1],emb.shape[-1])
              y_shaped=reshape_layer(y)
              emb+=y_shaped
            return x+emb
        else:
            return self.Conv1dTrans(x)

            

class down(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,emb_dim=28):
        super(down, self).__init__()
        self.conv1d = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride,padding)
        # # this is for t to be transformed 
        # self.emb_layer = nn.Sequential(
        #     nn.SiLU(),
        #     nn.Linear(
        #         emb_dim,  ## this should be the inital size of t
        #         out_channels
        #     ),
        # )

    def forward(self, x,t,y):
            x=self.conv1d(x) 
            emb = (t)[:, :, None].repeat(1, x.shape[-2], x.shape[-1])
            if y is not None:
              reshape_layer=nn.Linear(y.shape[-1],emb.shape[-1])
              y_shaped=reshape_layer(y)
              emb+=y_shaped
            return x+emb



class Noise2pulse(nn.Module):
    def __init__(self, model_size=50, ngpus=1, num_channels=7,
                 latent_dim=100, post_proc_filt_len=512,
                 verbose=False, upsample=True, num_classes=None,time_dim=1,device=device):
        super(Noise2pulse, self).__init__()
        self.ngpus = ngpus
        self.model_size = model_size  # d
        self.num_channels = num_channels  # c
        self.time_dim=time_dim
        self.device=device
        #self.latent_di = latent_dim
        #self.post_proc_filt_len = post_proc_filt_len
  

       #Down layers
        self.conv_1 = down(num_channels, 10, 25, stride=2, padding=25 // 2)
        self.conv_2 = down(10, 25, 25, stride=5, padding= 25 // 2)
        self.conv_3 = down(25, 50 , 25, stride=2, padding= 25 // 2)
        self.conv_4 = down(50, 150 , 25, stride=5, padding= 25 // 2)
        self.conv_5 = down(150, 250 , 25, stride=5, padding= 25 // 2)
        self.conv_6 = down(250, 250 , 25, stride=5, padding= 25 // 2)

        # Up layers
        stride = 4
        if upsample:
            stride = 1
            upsample = 5
        self.deconv_1 = up_first_last(250 , 250, 25, stride, upsample=upsample)
        self.deconv_2 = up(500, 150, 25, stride, upsample=upsample)
        self.deconv_3 = up(300,  50, 25, stride, upsample=upsample)
        self.deconv_4 = up(100, 25, 25, stride, upsample=2)
        self.deconv_5 = up( 50, 10, 25, stride, upsample=upsample)
        self.deconv_6 = up_first_last(10, 7, 25, stride, upsample=2)


      #   self.deconv_1 = Transpose1dLayer(5 * model_size , 5 * model_size, 25, stride, upsample=upsample)
      #   self.deconv_2 = Transpose1dLayer_multi_input(5 * model_size * 2, 3 * model_size, 25, stride, upsample=upsample)
      #   self.deconv_3 = Transpose1dLayer_multi_input(3 * model_size * 2,  model_size, 25, stride, upsample=upsample)
      #  # self.deconv_4 = Transpose1dLayer( model_size, model_size, 25, stride, upsample=upsample)
      #   self.deconv_5 = Transpose1dLayer_multi_input( model_size * 2, int(model_size / 2), 25, stride, upsample=2)
      #   self.deconv_6 = Transpose1dLayer_multi_input(  int(model_size / 2) * 2, int(model_size / 5), 25, stride, upsample=upsample)
      #   self.deconv_7 = Transpose1dLayer(  int(model_size / 5), num_channels, 25, stride, upsample=2)

#        if post_proc_filt_len:
#            self.ppfilter1 = nn.Conv1d(num_channels, num_channels, post_proc_filt_len)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight.data)

        #self.label_emb = nn.Embedding(num_classes, time_dim)

    def pos_encoding(self, t, channels):
        """
        takes in t and channels, returns pos_enc with shape (batch,channels)
        """
        # inv_freq = 1.0 / (
        #     10000
        #     ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        # )
        # pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        # pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        # pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        pos_enc = torch.sin(t)
        return pos_enc

    def forward(self,x,t,y):

        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)


        #if y is not None:
            #t += self.label_emb(y)
        #print("one")
        conv_1_out = F.leaky_relu(self.conv_1(x,t,y)) # x=1,7,5000 -- 1,10,2500
        #print("two",conv_1_out.shape)
        conv_2_out = F.leaky_relu(self.conv_2(conv_1_out,t,y)) #1,25,500
        #print("three",conv_2_out.shape)
        conv_3_out = F.leaky_relu(self.conv_3(conv_2_out,t,y)) #1,50,250
        #print("four",conv_3_out.shape)
        conv_4_out = F.leaky_relu(self.conv_4(conv_3_out,t,y)) # 1,150,50
        #print("5",conv_4_out.shape)
        conv_5_out = F.leaky_relu(self.conv_5(conv_4_out,t,y)) #1,250,10
        #print("6",conv_5_out.shape)
        conv_6_out = F.leaky_relu(self.conv_6(conv_5_out,t,y)) #1,250,2
        #print("7",conv_6_out.shape)

        deconv_1_out = F.relu(self.deconv_1(conv_6_out)) #1,250,10
        #print("8",deconv_1_out.shape)
        deconv_2_out = F.relu(self.deconv_2(deconv_1_out, conv_5_out,t,y))
        #print("9",deconv_2_out.shape)
        deconv_3_out = F.relu(self.deconv_3(deconv_2_out, conv_4_out,t,y))
        #print("10",deconv_3_out.shape)
        deconv_4_out = F.relu(self.deconv_4(deconv_3_out, conv_3_out,t,y))
        #print("11",deconv_4_out.shape)
        deconv_5_out = F.relu(self.deconv_5(deconv_4_out, conv_2_out,t,y))
        #print("12",deconv_5_out.shape)
        output =   torch.tanh(self.deconv_6(deconv_5_out))
        #print("13",output.shape)


        return output







