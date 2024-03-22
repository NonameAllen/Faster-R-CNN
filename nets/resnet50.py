import math

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

# RGA注意力机制
class RGA_Module(nn.Module):
    def __init__(self,in_channel,in_spatial,use_spatial=True,use_channel=True,
        cha_ratio=8,spa_ratio=8,down_ratio=8):
        super(RGA_Module,self).__init__()
        self.in_channel=in_channel  # C=256
        self.in_spatial=in_spatial  # H*W=64*32=2048
        self.inter_channel=in_channel//cha_ratio    # C//8=256//8=32
        self.inter_spatial=in_spatial//spa_ratio    # (H*W)//8=2048//8=256
        self.use_spatial=use_spatial    # 是否使用RGA-S
        self.use_channel=use_channel    # 是否使用RGA-C
        if self.use_spatial:
            # 定义5个卷积
            # (8*256*64*32)--(8*32*64*32)
            self.theta_spatial=nn.Sequential(
                    nn.Conv2d(in_channels=self.in_channel,
                            out_channels=self.inter_channel,
                            kernel_size=1,stride=1,padding=0,bias=False),
                    nn.BatchNorm2d(self.inter_channel),
                    nn.ReLU()
            )
            # (8*256*64*32)--(8*32*64*32)
            self.phi_spatial=nn.Sequential(
                    nn.Conv2d(in_channels=self.in_channel,
                            out_channels=self.inter_channel,
                            kernel_size=1,stride=1,padding=0,bias=False),
                    nn.BatchNorm2d(self.inter_channel),
                    nn.ReLU()
            )
            # (8*4096*64*32)--(8*256*64*32)
            self.gg_spatial=nn.Sequential(
                    nn.Conv2d(in_channels=self.in_spatial*2,
                            out_channels=self.inter_spatial,
                            kernel_size=1,stride=1,padding=0,bias=False),
                    nn.BatchNorm2d(self.inter_spatial),
                    nn.ReLU()
            )
            # (8*256*64*32)--(8*32*64*32)
            self.gx_spatial=nn.Sequential(
                    nn.Conv2d(in_channels=self.in_channel,
                            out_channels=self.inter_channel,
                            kernel_size=1,stride=1,padding=0,bias=False),
                    nn.BatchNorm2d(self.inter_channel),
                    nn.ReLU()
            )
            # (8*257*64*32)--(8*1*64*32)
            num_channel_s=1+self.inter_spatial
            self.W_spatial=nn.Sequential(
                    nn.Conv2d(in_channels=num_channel_s,
                            out_channels=num_channel_s//down_ratio,
                            kernel_size=1,stride=1,padding=0,bias=False),
                    nn.BatchNorm2d(num_channel_s//down_ratio),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=num_channel_s//down_ratio,
                            out_channels=1,
                            kernel_size=1,stride=1,padding=0,bias=False),
                    nn.BatchNorm2d(1)
            )
        if self.use_channel:
            # 定义5个卷积
            # (8*2048*256*1)--(8*256*256*1)
            self.theta_channel=nn.Sequential(
                    nn.Conv2d(in_channels=self.in_spatial,
                            out_channels=self.inter_spatial,
                            kernel_size=1,stride=1,padding=0,bias=False),
                    nn.BatchNorm2d(self.inter_spatial),
                    nn.ReLU()
            )
            # (8*2048*256*1)--(8*256*256*1)
            self.phi_channel=nn.Sequential(
                    nn.Conv2d(in_channels=self.in_spatial,
                            out_channels=self.inter_spatial,
                            kernel_size=1,stride=1,padding=0,bias=False),
                    nn.BatchNorm2d(self.inter_spatial),
                    nn.ReLU()
            )
            # (8*512*256*1)--(8*32*256*1)
            self.gg_channel=nn.Sequential(
                    nn.Conv2d(in_channels=self.in_channel*2,
                            out_channels=self.inter_channel,
                            kernel_size=1,stride=1,padding=0,bias=False),
                    nn.BatchNorm2d(self.inter_channel),
                    nn.ReLU()
            )
            # (8*2048*256*1)--(8*256*256*1)
            self.gx_channel=nn.Sequential(
                    nn.Conv2d(in_channels=self.in_spatial,
                            out_channels=self.inter_spatial,
                            kernel_size=1,stride=1,padding=0,bias=False),
                    nn.BatchNorm2d(self.inter_spatial),
                    nn.ReLU()
            )
            # (8*33*256*1)--(8*1*256*1)
            num_channel_c=1+self.inter_channel
            self.W_channel=nn.Sequential(
                    nn.Conv2d(in_channels=num_channel_c,
                            out_channels=num_channel_c//down_ratio,
                            kernel_size=1,stride=1,padding=0,bias=False),
                    nn.BatchNorm2d(num_channel_c//down_ratio),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=num_channel_c//down_ratio,
                            out_channels=1,
                            kernel_size=1,stride=1,padding=0,bias=False),
                    nn.BatchNorm2d(1)
            )
    def forward(self,x):
        # input x:(8,256,64,32)
        b,c,h,w=x.size()    # 8,256,64,32
        if self.use_spatial:
            theta_xs=self.theta_spatial(x)  # 8*32*64*32
            phi_xs=self.phi_spatial(x)      # 8*32*64*32
            theta_xs=theta_xs.view(b,self.inter_channel,-1) # 8*32*2048
            theta_xs=theta_xs.permute(0,2,1)                # 8*2048*32
            phi_xs=phi_xs.view(b,self.inter_channel,-1)     # 8*32*2048
            Gs=torch.matmul(theta_xs,phi_xs)                # 8*2048*2048
            Gs_in=Gs.permute(0,2,1).view(b,h*w,h,w)         # 8*2048*64*32
            Gs_out=Gs.view(b,h*w,h,w)                       # 8*2048*64*32
            Gs_joint=torch.cat((Gs_in,Gs_out),1)            # 8*4096*64*32
            Gs_joint=self.gg_spatial(Gs_joint)              # 8*256*64*32
            g_xs=self.gx_spatial(x)                         # 8*32*64*32
            g_xs=torch.mean(g_xs,dim=1,keepdim=True)        # 8*1*64*32
            ys=torch.cat((g_xs,Gs_joint),1)                 # 8*257*64*32
            w_ys=self.W_spatial(ys)                         # 8*1*64*32
            if not self.use_channel:
                out=torch.sigmoid(w_ys.expand_as(x))*x      # 8*256*64*32
                return out
            else:
                x=torch.sigmoid(w_ys.expand_as(x))*x        # 8*256*64*32
        if self.use_channel:
            xc=x.view(b,c,-1).permute(0,2,1).unsqueeze(-1)              # 8*2048*256*1
            theta_xc=self.theta_channel(xc).squeeze(-1).permute(0,2,1)  # 8*256*256
            phi_xc=self.phi_channel(xc).squeeze(-1) # 8*256*256
            Gc=torch.matmul(theta_xc,phi_xc)        # 8*256*256
            Gc_in=Gc.permute(0,2,1).unsqueeze(-1)   # 8*256*256*1
            Gc_out=Gc.unsqueeze(-1)                 # 8*256*256*1
            Gc_joint=torch.cat((Gc_in,Gc_out),1)    # 8*512*256*1
            Gc_joint=self.gg_channel(Gc_joint)      # 8*32*256*1
            g_xc=self.gx_channel(xc)                # 8*256*256*1
            g_xc=torch.mean(g_xc,dim=1,keepdim=True)    # 8*1*256*1
            yc=torch.cat((g_xc,Gc_joint),1)             # 8*33*256*1
            w_yc=self.W_channel(yc).transpose(1,2)      # 8*1*256*1--8*256*1*1
            out=torch.sigmoid(w_yc)*x                   # 8*256*64*32
            return out
        
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet_RGA(nn.Module):
    def __init__(self, block, layers, num_classes=1000,height=256,width=128,spa_on=True,
                    cha_on=True,s_ratio=8,c_ratio=8,d_ratio=8 ):
        #-----------------------------------#
        #   假设输入进来的图片是600,600,3
        #-----------------------------------#
        self.inplanes = 64
        super(ResNet_RGA, self).__init__()

        # 600,600,3 -> 300,300,64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 300,300,64 -> 150,150,64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        # 150,150,64 -> 150,150,256
        self.layer1 = self._make_layer(block, 64, layers[0])
        # 150,150,256 -> 75,75,512
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # 75,75,512 -> 38,38,1024 到这里可以获得一个38,38,1024的共享特征层
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4被用在classifier模型中
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
         # RGA注意力机制
        self.rga1=RGA_Module(256,(height//4)*(width//4),use_spatial=spa_on,use_channel=cha_on,
                                cha_ratio=c_ratio,spa_ratio=s_ratio,down_ratio=d_ratio)
        self.rga2=RGA_Module(512,(height//8)*(width//8),use_spatial=spa_on,use_channel=cha_on,
                                cha_ratio=c_ratio,spa_ratio=s_ratio,down_ratio=d_ratio)
        self.rga3=RGA_Module(1024,(height//16)*(width//16),use_spatial=spa_on,use_channel=cha_on,
                                cha_ratio=c_ratio,spa_ratio=s_ratio,down_ratio=d_ratio)
        self.rga4=RGA_Module(2048,(height//16)*(width//16),use_spatial=spa_on,use_channel=cha_on,
                                cha_ratio=c_ratio,spa_ratio=s_ratio,down_ratio=d_ratio)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        #-------------------------------------------------------------------#
        #   当模型需要进行高和宽的压缩的时候，就需要用到残差边的downsample
        #-------------------------------------------------------------------#
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def resnet50(pretrained = False):
    model = ResNet_RGA(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/resnet50-19c8e357.pth", model_dir="./model_data")
        model.load_state_dict(state_dict)
    #----------------------------------------------------------------------------#
    #   获取特征提取部分，从conv1到model.layer3，最终获得一个38,38,1024的特征层
    #----------------------------------------------------------------------------#
    features    = list([model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3])
    #----------------------------------------------------------------------------#
    #   获取分类部分，从model.layer4到model.avgpool
    #----------------------------------------------------------------------------#
    classifier  = list([model.layer4, model.avgpool])
    
    features    = nn.Sequential(*features)
    classifier  = nn.Sequential(*classifier)
    return features, classifier
