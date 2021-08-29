import torch
import torch.nn as nn
import numpy as np

class ResidualPointNetBlock(nn.Module):
    def __init__(self,in_dim,inter_dim,out_dim):
        super().__init__()
        self.conv_in=nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.ReLU(),
            nn.Conv2d(in_dim,inter_dim,1,1),
            nn.BatchNorm2d(inter_dim),
            nn.ReLU(),
            nn.Conv2d(inter_dim,inter_dim,1,1)
        )
        self.conv_out=nn.Sequential(
            nn.Conv2d(inter_dim*2,out_dim,1,1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Conv2d(out_dim,out_dim,1,1),
        )
        self.short_cut=None
        if in_dim!=out_dim:
            self.short_cut=nn.Sequential(
                nn.BatchNorm2d(in_dim),
                nn.ReLU(),
                nn.Conv2d(in_dim,out_dim,1,1)
            )

    def forward(self,feats):
        """
        :param feats: b,f,k,n
        :return:
        """
        nn=feats.shape[-1]
        feats_in=self.conv_in(feats)
        feats_pool=torch.max(feats_in,3,keepdim=True)[0]
        feats_pool=feats_pool.repeat(1,1,1,nn)
        feats_out=self.conv_out(torch.cat([feats_in,feats_pool],1))
        if self.short_cut is None:
            feats_out=feats_out+feats
        else:
            feats_out=feats_out+self.short_cut(feats)
        return feats_out


class PN(nn.Module):
    def __init__(self,radius):
        super().__init__()
        self.r=radius
        self.conv_in=nn.Conv2d(3,64,1,1)
        self.conv_mid=nn.ModuleList([ResidualPointNetBlock(64,64,64),ResidualPointNetBlock(64,64,64)])
        self.conv_out=nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,32,1,1)
        )
    
    def forward(self,input):
        center = input[:, -1, :].unsqueeze(1)
        delta_x = input[:, :, 0:3] - center[:, :, 0:3]  # (B, npoint, 3), normalized coordinates
        delta_x = delta_x/self.r
        delta_x = delta_x.permute(0,2,1)[:,:,:,None]  # (B, 3, npoint, 1)
        data = self.conv_in(delta_x)
        for layer in self.conv_mid:
            data=layer(data)
        data = self.conv_out(data)
        data = torch.max(data,dim=2)[0][:,:,None,:]
        #data=self.conv_final(data)[:,:,0,0]
        return data # (B, 32)
    
    def get_parameter(self):
        return list(self.parameters())
