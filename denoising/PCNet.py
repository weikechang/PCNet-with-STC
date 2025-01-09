import torch
import torch.nn as nn

#==================Unet utils====================
class conv(nn.Module):
    def __init__(self, inC, outC, kernel_size, padding, momentum):
        super(conv, self).__init__()
        self.c = nn.Sequential(nn.Conv3d(inC, outC, kernel_size=kernel_size, padding=padding),
                               nn.BatchNorm3d(outC,momentum=momentum),
                               nn.ReLU(inplace=True))    
    def forward(self, x):
        x = self.c(x)
        return x

class convT(nn.Module):
    def __init__(self, inC, outC, kernel_size, padding,output_padding, stride, momentum):
        super(convT, self).__init__()
        self.cT = nn.Sequential(nn.ConvTranspose3d(inC, outC, kernel_size=kernel_size, 
                                                   padding=padding,output_padding=output_padding,
                                                   stride=stride),
                                nn.ReLU(inplace=True))
    def forward(self, x):
        x = self.cT(x)
        return x

class double_conv(nn.Module):
  #conv --> BN --> ReLUx2
    def __init__(self, inC, outC, kernel_size, padding, momentum):
        super(double_conv, self).__init__()
        self.conv2x = nn.Sequential(
            conv(inC, outC, kernel_size=kernel_size, padding=padding, momentum=momentum),
            conv(outC, outC, kernel_size=kernel_size, padding=padding, momentum=momentum))
   
    def forward(self, x):
        x = self.conv2x(x)
        return x

class down(nn.Module):
    def __init__(self, inC, outC, momentum):
        super(down, self).__init__()
        #go down = maxpool + double conv nn.MaxPool(2)             nn.Conv2d(inC, outC, kernel_size=2, stride=2, padding=0)
        self.go_down = nn.Sequential(
            nn.MaxPool3d(2),
            double_conv(inC, outC, kernel_size=3, padding=1, momentum=momentum))   
    def forward(self, x):
        x = self.go_down(x)
        return x

class up(nn.Module):
    def __init__(self, inC, outC, momentum):
        super(up, self).__init__()
        #go up = conv2d to half C-->upsample
        self.convt1 = convT(inC, outC, kernel_size=3, padding=1,output_padding=1, stride=2, momentum=momentum)
        self.conv2x = double_conv(inC, outC, kernel_size=3, padding=1, momentum=momentum)
    
    def forward(self, x1, x2):
        #x1 is data from a previous layer, x2 is current input
        x2 = self.convt1(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv2x(x)
        return x

class outconv(nn.Module):
    def __init__(self, inC, outC, kernel_size, padding):
        super(outconv, self).__init__()
        self.conv = nn.Conv3d(in_channels=inC, out_channels=outC, kernel_size=kernel_size, padding=padding, bias=False)
    def forward(self, x):
        x = self.conv(x)
        return x
    
#=================Unet=====================
class Unet(nn.Module):
    def __init__(self,C_size = 32,momentum=0.1):
        super(Unet, self).__init__()
        inC=1
        outC=1
        self.inc = conv(inC, C_size, 3, 1, momentum=momentum)
        self.down1 = down(C_size, C_size*2, momentum)
        self.down2 = down(C_size*2, C_size*4, momentum)
        self.down3 = down(C_size*4, C_size*8, momentum)

        self.up1 = up(C_size*8, C_size*4, momentum)
        self.up2 = up(C_size*4, C_size*2, momentum)
        self.up3 = convT(C_size*2, C_size, kernel_size=3, padding=1,output_padding=1, stride=2, momentum=momentum)
        self.up3_conv = conv(C_size*2, C_size, 3, 1, momentum=momentum)
        self.outc = outconv(C_size, outC, 3, 1)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x0 = self.up1(x3, x4)
        x0 = self.up2(x2, x0)
        x0 = self.up3(x0)
        x0 = torch.cat([x0, x1], dim=1)
        x0 = self.up3_conv(x0)
        x0 = self.outc(x0)
        return x - x0 
        
class GD_CNN(nn.Module):
    
    def __init__(self):
        super(GD_CNN, self).__init__()


        self.R = Unet()
                             
    def forward(self, alpha, beta, x, b):
        output = x - b
        
        output_v = self.R(x)
        output = self.mid( alpha, beta, output, x, output_v)
        
        return output
        
    def mid(self, alpha, beta, output, x, output_v):
        output = x - torch.mul(alpha,(output+torch.mul(beta,(x-output_v)))) 
        
        return output
    
class MEM_CNN(nn.Module):
    
    def __init__(self):
        super(MEM_CNN, self).__init__()
       
        self.delta = GD_CNN()
        self.D = Unet()
        
        # Defining learnable parameters
        self.alpha_0 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.beta_0 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.h_0 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        
        self.alpha_1 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.beta_1 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.h_1 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        
        self.alpha_0.data = torch.tensor(0.1)
        self.beta_0.data = torch.tensor(0.9)
        self.h_0.data = torch.tensor(1.5)
        
        self.alpha_1.data = torch.tensor(0.1)
        self.beta_1.data = torch.tensor(0.9)
        self.h_1.data = torch.tensor(1.5)
        
        
    def forward(self, input):
        x = input.clone().detach()
        b = input.clone().detach()
        #K = torch.cat((x, torch.FloatTensor([10 / 255.]).repeat(1,1,x.shape[2],x.shape[3])),dim = 1)
        #x = torch.zeros(b.shape, dtype = b.dtype, layout = b.layout, device = b.device)
        for i in range(2):
            if i == 0:
                alpha = self.alpha_0
                beta = self.beta_0
                h = self.h_0
            if i == 1:
                alpha = self.alpha_1
                beta = self.beta_1
                h = self.h_1 
            delta_x = self.delta(alpha, beta, x, b)
            output = delta_x - b
        
            output_delta_v = self.D(delta_x)
        
            outputs = self.mid_delta(alpha, beta, x, output, delta_x, output_delta_v)
            x = self.final(outputs,delta_x,h)
        
        return x
            
    def mid_delta(self, alpha, beta, x, output, delta_x, output_delta_v):
        outputs = x + torch.mul(alpha,(output+torch.mul(beta,(delta_x-output_delta_v)))) 
        
        return outputs
    
    def final(self,outputs, delta_x, h):
        
        return torch.mul(1+h/2,delta_x) - torch.mul(h/2,outputs)        
