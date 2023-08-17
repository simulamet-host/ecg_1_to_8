import torch
import torch.nn.functional as F


class Upscale1dLayer(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, scale_factor=None, linear_interpolation: bool = True):
        super(Upscale1dLayer, self).__init__()

        if linear_interpolation:
            self.upsample_layer = torch.nn.Upsample(scale_factor=scale_factor, mode='linear', align_corners=True)
        else:
            self.upsample_layer = torch.nn.Upsample(scale_factor=scale_factor)
        self.conv1d = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding = kernel_size // 2, padding_mode='reflect') 

    def forward(self, x):
        return self.conv1d(self.upsample_layer(x))


class Upscale1dLayer_multi_input(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, scale_factor=None, linear_interpolation: bool = True):
        super(Upscale1dLayer_multi_input, self).__init__()

        if linear_interpolation:
            self.upsample_layer = torch.nn.Upsample(scale_factor=scale_factor, mode='linear', align_corners=True)
        else:
            self.upsample_layer = torch.nn.Upsample(scale_factor=scale_factor)
        self.conv1d = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding = kernel_size // 2, padding_mode='reflect')

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        return self.conv1d(self.upsample_layer(x))


class EcgUNetGenerator(torch.nn.Module):
    def __init__(self, model_size: int = 32, num_input_channels: int = 1, num_output_channels: int = 7, kernel_size: int = 25, verbose: bool = False):
        super(EcgUNetGenerator, self).__init__()
        self.verbose = verbose

        padding_size = kernel_size // 2
        
        self.conv_1 = torch.nn.Conv1d(num_input_channels, model_size, kernel_size, stride=2, padding = padding_size, padding_mode='reflect')
        self.conv_2 = torch.nn.Conv1d(model_size, model_size * 2, kernel_size, stride=2, padding = padding_size, padding_mode='reflect')
        self.conv_3 = torch.nn.Conv1d(model_size * 2, model_size * 4, kernel_size, stride=2, padding = padding_size, padding_mode='reflect') 
        self.conv_4 = torch.nn.Conv1d(model_size * 4, model_size * 8, kernel_size, stride=5, padding = padding_size, padding_mode='reflect')
        self.conv_5 = torch.nn.Conv1d(model_size * 8, model_size * 16, kernel_size, stride=5, padding = padding_size, padding_mode='reflect')
        self.conv_6 = torch.nn.Conv1d(model_size * 16, model_size * 16, kernel_size, stride=5, padding = padding_size, padding_mode='reflect')
        
        self.dropout1 = torch.nn.Dropout(0.1)
        self.deconv_1 = Upscale1dLayer(16 * model_size, 16 * model_size, kernel_size, stride=1, scale_factor=5)
        self.dropout2 = torch.nn.Dropout(0.1)
        self.deconv_2 = Upscale1dLayer_multi_input(32 * model_size, 8 * model_size, kernel_size, stride=1, scale_factor=5)
        self.dropout3 = torch.nn.Dropout(0.1)
        self.deconv_3 = Upscale1dLayer_multi_input(16 * model_size, 4 * model_size, kernel_size, stride=1, scale_factor=5)
        self.deconv_4 = Upscale1dLayer_multi_input(8 * model_size, 2 * model_size, kernel_size, stride=1, scale_factor=2)
        self.deconv_5 = Upscale1dLayer_multi_input(4 * model_size, model_size, kernel_size, stride=1, scale_factor=2)
        self.deconv_6 = Upscale1dLayer(model_size, num_output_channels, kernel_size, stride=1, scale_factor=2)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        if self.verbose: print(f'X: {x.shape}')
        conv_1_out = F.leaky_relu(self.conv_1(x))
        if self.verbose: print(f'Conv1: {conv_1_out.shape}')
        conv_2_out = F.leaky_relu(self.conv_2(conv_1_out))
        if self.verbose: print(f'Conv2: {conv_2_out.shape}')
        conv_3_out = F.leaky_relu(self.conv_3(conv_2_out))
        if self.verbose: print(f'Conv3: {conv_3_out.shape}')
        conv_4_out = F.leaky_relu(self.conv_4(conv_3_out))
        if self.verbose: print(f'Conv4: {conv_4_out.shape}')
        conv_5_out = F.leaky_relu(self.conv_5(conv_4_out))
        if self.verbose: print(f'Conv5: {conv_5_out.shape}')
        conv_6_out = F.leaky_relu(self.dropout1(self.conv_6(conv_5_out)))
        if self.verbose: print(f'Conv6: {conv_6_out.shape}')
  
        deconv_1_out = F.relu(self.dropout2(self.deconv_1(conv_6_out)))
        if self.verbose: print(f'Deconv1: {deconv_1_out.shape}')
        deconv_2_out = F.relu(self.dropout3(self.deconv_2(deconv_1_out, conv_5_out)))
        if self.verbose: print(f'Deconv2: {deconv_2_out.shape}')
        deconv_3_out = F.relu(self.deconv_3(deconv_2_out, conv_4_out))
        if self.verbose: print(f'Deconv3: {deconv_3_out.shape}')
        deconv_4_out = F.relu(self.deconv_4(deconv_3_out, conv_3_out))
        if self.verbose: print(f'Deconv4: {deconv_4_out.shape}')
        deconv_5_out = F.relu(self.deconv_5(deconv_4_out, conv_2_out))
        if self.verbose: print(f'Deconv5: {deconv_5_out.shape}')
        deconv_6_out = self.deconv_6(deconv_5_out)
        if self.verbose: print(f'Deconv6: {deconv_6_out.shape}')

        output = torch.tanh(deconv_6_out)
        if self.verbose: print(output.shape)

        return output


class EcgGanDiscriminator(torch.nn.Module):
    def __init__(self, model_size: int = 32, num_channels: int = 8, kernel_size: int = 13, alpha: float = 0.2, verbose: bool = False):
        super(EcgGanDiscriminator, self).__init__()
        self.alpha = alpha
        self.verbose = verbose
        
        padding_size = kernel_size // 2
        self.conv1 = torch.nn.Conv1d(num_channels,  model_size, kernel_size, stride=2, padding=padding_size, padding_mode='reflect')
        self.conv2 = torch.nn.Conv1d(model_size, 2 * model_size, kernel_size, stride=2, padding=padding_size, padding_mode='reflect')
        self.conv3 = torch.nn.Conv1d(2 * model_size, 4 * model_size, kernel_size, stride=2, padding=padding_size, padding_mode='reflect')
        self.conv4 = torch.nn.Conv1d(4 * model_size, 8 * model_size, kernel_size, stride=2, padding=padding_size, padding_mode='reflect')
        self.conv5 = torch.nn.Conv1d(8 * model_size, 16 * model_size, kernel_size, stride=2, padding=padding_size, padding_mode='reflect')
        self.conv6 = torch.nn.Conv1d(16 * model_size, 16 * model_size, kernel_size, stride=2, padding=padding_size, padding_mode='reflect')
        self.conv7 = torch.nn.Conv1d(16 * model_size, 16 * model_size, kernel_size, stride=4, padding=padding_size, padding_mode='reflect')

        self.linear = torch.nn.Linear(2048, 1)
        self.activ = torch.nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        if self.verbose: print(f'x: {x.shape}')
        conv_1_out = F.leaky_relu(self.conv1(x), negative_slope=self.alpha)
        if self.verbose: print(f'conv_1_out: {conv_1_out.shape}')
        conv_2_out = F.leaky_relu(self.conv2(conv_1_out), negative_slope=self.alpha)
        if self.verbose: print(f'conv_2_out: {conv_2_out.shape}')
        conv_3_out = F.leaky_relu(self.conv3(conv_2_out), negative_slope=self.alpha)
        if self.verbose: print(f'conv_3_out: {conv_3_out.shape}')
        conv_4_out = F.leaky_relu(self.conv4(conv_3_out), negative_slope=self.alpha)
        if self.verbose: print(f'conv_4_out: {conv_4_out.shape}')
        conv_5_out = F.leaky_relu(self.conv5(conv_4_out), negative_slope=self.alpha)
        if self.verbose: print(f'conv_5_out: {conv_5_out.shape}')
        conv_6_out = F.leaky_relu(self.conv6(conv_5_out), negative_slope=self.alpha)
        if self.verbose: print(f'conv_6_out: {conv_6_out.shape}')
        conv_7_out = F.leaky_relu(self.conv7(conv_6_out), negative_slope=self.alpha)
        if self.verbose: print(f'conv_7_out: {conv_7_out.shape}')
        conv_7_view_out = conv_7_out.view(-1, conv_7_out.shape[1] * conv_7_out.shape[2])
        if self.verbose: print(f'conv_7_view_out: {conv_7_view_out.shape}')
        
        fc_out = self.linear(conv_7_view_out)
        # return fc_out # use this instead of the sigmoid if you plan to replace MSE with BCE for GAN Loss
        return self.activ(fc_out)
