from torch import nn, sqrt
import torch
import sys
from math import sqrt
sys.path.append('.')
from MBConv import MBConvBlock
from SelfAttention import ScaledDotProductAttention

class CoAtNet(nn.Module):
    def __init__(self,in_ch,image_size,out_chs=[64,96,192,384,768]):
        super().__init__()
        self.out_chs=out_chs
        self.maxpool2d=nn.MaxPool2d(kernel_size=2,stride=2)
        self.maxpool1d = nn.MaxPool1d(kernel_size=2, stride=2)

        self.s0=nn.Sequential(
            nn.Conv2d(in_ch,in_ch,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_ch,in_ch,kernel_size=3,padding=1)
        )
        self.mlp0=nn.Sequential(
            nn.Conv2d(in_ch,out_chs[0],kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_chs[0],out_chs[0],kernel_size=1)
        )
        
        self.s1=MBConvBlock(ksize=3,input_filters=out_chs[0],output_filters=out_chs[0],image_size=image_size//2)
        self.mlp1=nn.Sequential(
            nn.Conv2d(out_chs[0],out_chs[1],kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_chs[1],out_chs[1],kernel_size=1)
        )

        self.s2=MBConvBlock(ksize=3,input_filters=out_chs[1],output_filters=out_chs[1],image_size=image_size//4)
        self.mlp2=nn.Sequential(
            nn.Conv2d(out_chs[1],out_chs[2],kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_chs[2],out_chs[2],kernel_size=1)
        )

        self.s3=ScaledDotProductAttention(out_chs[2],out_chs[2]//8,out_chs[2]//8,8)
        self.mlp3=nn.Sequential(
            nn.Linear(out_chs[2],out_chs[3]),
            nn.ReLU(),
            nn.Linear(out_chs[3],out_chs[3])
        )

        self.s4=ScaledDotProductAttention(out_chs[3],out_chs[3]//8,out_chs[3]//8,8)
        self.mlp4=nn.Sequential(
            nn.Linear(out_chs[3],out_chs[4]),
            nn.ReLU(),
            nn.Linear(out_chs[4],out_chs[4])
        )
        
        self.fc = nn.Linear(in_features=768, out_features=1000)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x) :
        B,C,H,W=x.shape
        #stage0
        x=self.mlp0(self.s0(x))
        x=self.maxpool2d(x)
        #stage1
        x=self.mlp1(self.s1(x))
        x=self.maxpool2d(x)
        #stage2
        x=self.mlp2(self.s2(x))
        x=self.maxpool2d(x)
        #stage3
        x=x.reshape(B,self.out_chs[2],-1).permute(0,2,1) #B,N,C
        x=self.mlp3(self.s3(x,x,x))
        x=self.maxpool1d(x.permute(0,2,1)).permute(0,2,1)
        #stage4
        x=self.mlp4(self.s4(x,x,x))
        x=self.maxpool1d(x.permute(0,2,1))
        N=x.shape[-1]
        x=x.reshape(B,self.out_chs[4],int(sqrt(N)),int(sqrt(N)))
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.softmax(x)

        return x

if __name__ == '__main__':
    x=torch.randn(1,3,224,224)
    coatnet=CoAtNet(3,224)
    y=coatnet(x)
    print(y.shape)