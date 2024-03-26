from torch import nn
import torch
from .attention import PAM_Module, CAM_Module, Non_Local_Module
from advent.utils.func import bce_loss_trans


def get_fc_discriminator(num_classes, ndf=64):
    return nn.Sequential(
        nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1),
    )


def get_fc_pix_discriminator(num_classes, ndf=64):
    print('pixel level disc')
    return nn.Sequential(
        nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1),
        nn.ConvTranspose2d(ndf*8, 1, kernel_size=4, stride=2, padding=1),
        nn.Upsample(scale_factor=8, mode='bilinear'),
    )


def get_fc_pix_discriminator_3x3_1x1(num_classes, ndf=64, factor = 1):
    return nn.Sequential(
        nn.Conv2d(num_classes, ndf*factor, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * factor, ndf * 2 * factor, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 2 * factor, ndf * 4 * factor, kernel_size=1, stride=1, padding=0),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 4 * factor, ndf * 8 * factor, kernel_size=1, stride=1, padding=0),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 8 * factor, 1, kernel_size=1, stride=1, padding=0),
    )

def get_fc_pix_discriminator_1x1(num_classes, ndf=64, factor = 1):
    return nn.Sequential(
        nn.Conv2d(num_classes, ndf*factor, kernel_size=1, stride=1, padding=0),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * factor, ndf * 2 * factor, kernel_size=1, stride=1, padding=0),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 2 * factor, ndf * 4 * factor, kernel_size=1, stride=1, padding=0),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 4 * factor, ndf * 8 * factor, kernel_size=1, stride=1, padding=0),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 8 * factor, 1, kernel_size=1, stride=1, padding=0),
    )

def get_fc_pix_discriminatorBulk_1x1(num_classes, ndf=64, factor = 1):
    return nn.Sequential(
        nn.Conv2d(num_classes, ndf*factor, kernel_size=1, stride=1, padding=0),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * factor, ndf * 2 * factor, kernel_size=1, stride=1, padding=0),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 2 *factor, ndf * 2 * factor, kernel_size=1, stride=1, padding=0),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 2 * factor, ndf * 4 * factor, kernel_size=1, stride=1, padding=0),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 4 * factor, ndf * 4 * factor, kernel_size=1, stride=1, padding=0),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 4 * factor, ndf * 8 * factor, kernel_size=1, stride=1, padding=0),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 8 * factor, ndf * 8 * factor, kernel_size=1, stride=1, padding=0),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 8 * factor, 1, kernel_size=1, stride=1, padding=0),
    )

def get_fc_pix_discriminator_1x1_param(num_classes, ndf=64):
    return nn.Sequential(
        nn.Conv2d(num_classes, ndf, kernel_size=2, stride=2, padding=0),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 2, ndf * 4, kernel_size=1, stride=1, padding=0),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 4, ndf * 8, kernel_size=1, stride=1, padding=0),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 8, 1, kernel_size=1, stride=1, padding=0),
    )

def get_fc_pix_discriminator_1x1_PAM(num_classes, ndf=64):
    return nn.Sequential(
        PAM_Module(in_dim=num_classes),
        get_fc_pix_discriminator_1x1(num_classes,ndf)
    )

def get_fc_pix_discriminator_1x1_CAM(num_classes, ndf=64):
    return nn.Sequential(
        CAM_Module(in_dim=num_classes),
        get_fc_pix_discriminator_1x1(num_classes,ndf)
    )

def get_fc_pix_discriminator_1x1_Non_local(num_classes, ndf=64):
    return nn.Sequential(
        Non_Local_Module(in_dim=num_classes),
        get_fc_pix_discriminator_1x1(num_classes,ndf)
    )

class get_fc_pix_discriminator_1x1_cat(nn.Module):
    def __init__(self, num_classes, ndf=64) :
        super(get_fc_pix_discriminator_1x1_cat, self).__init__()
        self.num_classes = num_classes
        factor = 1
        self.disc_1 = nn.Sequential(
        nn.Conv2d(num_classes, ndf*factor, kernel_size=1, stride=1, padding=0),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * factor, ndf * 2 * factor, kernel_size=1, stride=1, padding=0),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 2 * factor, ndf * 4 * factor, kernel_size=1, stride=1, padding=0),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 4 * factor, ndf * 8 * factor, kernel_size=1, stride=1, padding=0),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        
    )
        self.disc_d = nn.Conv2d(ndf * 8 * factor, 1, kernel_size=1, stride=1, padding=0)
        self.disc_cat = nn.Conv2d(ndf * 8 * factor, num_classes, kernel_size=1, stride=1, padding=0)
    

    def forward(self,x) :
        inter = self.disc_1(x)
        return self.disc_cat(inter), self.disc_d(inter)

class get_fc_pix_discriminator_1x1_Non_local_2(nn.Module):
    def __init__(self, num_classes, height, width,ndf=64) :
        super(get_fc_pix_discriminator_1x1_Non_local_2, self).__init__()
        self.num_classes = num_classes
        self.disc = get_fc_pix_discriminator_1x1(num_classes,ndf)
        self.Non_Local_Module = Non_Local_Module(in_dim=num_classes)
        self.height = height
        self.width = width

    def forward(self,x ) :
        print(x.shape)
        x = self.Non_Local_Module(x)
        x = torch.nn.functional.upsample(x, size=(self.height, self.width), mode='bilinear',align_corners=True)
        return self.disc(x)


class get_fc_pix_discriminator_1x1_CAM_right(nn.Module):
    def __init__(self, num_classes, ndf=64) :
        super(get_fc_pix_discriminator_1x1_CAM_right, self).__init__()
        self.num_classes = num_classes
        self.disc = get_fc_pix_discriminator_1x1(num_classes,ndf)
        self.CAM_Module = CAM_Module(in_dim=num_classes)

    def forward(self,x , isgen) :
        if isgen is False :
            #print(f' isgen False .. disc')
            x = self.CAM_Module(x)
        return self.disc(x)

class get_fc_pix_discriminator_1x1_solo(nn.Module):
    def __init__(self, num_classes, ndf=64) :
        super(get_fc_pix_discriminator_1x1_solo, self).__init__()
        print('solo')
        self.num_classes = num_classes
        self.disc = get_fc_pix_discriminator_1x1(num_classes,ndf)

    def forward(self, x, lbl) :
        m_batchsize, C, height, width = x.size()
        loss = 0
        x = x.permute(2,3,0,1)
        if type(lbl) is int :
            lbl_val = lbl
            lbl = torch.FloatTensor(torch.Size([m_batchsize,height, width]))
            lbl.fill_(lbl_val)
            lbl = lbl.permute(1,2,0)
        elif len(lbl.size()) == 4 :
            lbl = lbl.permute(2,3,0,1)
        else :
            lbl = lbl.permute(1,2,0)
        for i in range(0,height ) :
            for j in range(0, width) :
                if i==0 and j ==0 :
                    print(x[i][j].size())
                    print(self.disc(x[i][j].unsqueeze(2).unsqueeze(3)).size())
                    print(lbl[i][j].size())
                    loss = bce_loss_trans(self.disc(x[i][j].unsqueeze(2).unsqueeze(3)).squeeze(3).squeeze(2),lbl[i][j])
                else :
                    loss += bce_loss_trans(self.disc(x[i][j].unsqueeze(2).unsqueeze(3)).squeeze(3).squeeze(2), lbl[i][j])

        return loss/(height*width*1.0)

