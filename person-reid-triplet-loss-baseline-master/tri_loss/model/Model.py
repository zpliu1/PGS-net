import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from .resnet import resnet50 as resnet51
import scipy.io
from torch.nn import Conv2d, Parameter, Softmax, Dropout
from torch.nn.utils.weight_norm import weight_norm

from visdom import Visdom
import argparse

from tensorboardX import SummaryWriter
writer = SummaryWriter()

from torchvision.models.resnet import resnet50, Bottleneck

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Model(nn.Module):
  def __init__(self, last_conv_stride=2):
    super(Model, self).__init__()
    #self.base = resnet51(pretrained=True, last_conv_stride=last_conv_stride)

    resnet = resnet50(pretrained=True)

    self.backone = nn.Sequential(
        resnet.conv1,
        resnet.bn1,
        resnet.relu,
        resnet.maxpool,
        resnet.layer1,
        resnet.layer2,
        #resnet.layer3[0],
    )

    res_conv3 = resnet.layer3[0]
    res_conv3_1 = resnet.layer3[1]

    res_conv4 = nn.Sequential(*resnet.layer3[2:])

    res_p_conv5 = nn.Sequential(
        Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
        Bottleneck(2048, 512),
        Bottleneck(2048, 512))
    res_p_conv5.load_state_dict(resnet.layer4.state_dict())

    self.p00 = nn.Sequential(copy.deepcopy(res_conv3), copy.deepcopy(res_conv3_1))
    self.p0 = copy.deepcopy(res_conv4)
    self.p1 = copy.deepcopy(res_p_conv5)
    #self.p2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))

    self.Y0 = copy.deepcopy(res_conv4)
    self.Y1 = copy.deepcopy(res_p_conv5)

    '''
    self.backone = nn.Sequential(
        resnet.conv1,
        resnet.bn1,
        resnet.relu,
        resnet.maxpool,
        resnet.layer1,
        resnet.layer2,
        resnet.layer3[0],
    )

    res_conv4 = nn.Sequential(*resnet.layer3[1:])

    res_p_conv5 = nn.Sequential(
        Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
        Bottleneck(2048, 512),
        Bottleneck(2048, 512))
    res_p_conv5.load_state_dict(resnet.layer4.state_dict())

    self.p1 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
    '''
    #self.p2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
    #res_g_conv5 = resnet.layer4
    #self.part_2 = Bottleneck(2048, 512)
    #self.part_2.load_state_dict(resnet.layer4[2].state_dict())

    #self.bo1 = Bottleneck(2048, 512)
    #self.bo2 = Bottleneck(2048, 512)
    '''
    self.classifer3 = torch.nn.Sequential(
        torch.nn.Linear(2048, 6144),
        torch.nn.BatchNorm1d(6144),
        # torch.nn.ReLU(),  # jia(11.8)
        # torch.nn.Dropout(p=0.5),           #try 0.75
        torch.nn.Linear(6144, 1502))
    '''
    self.classifer1 = torch.nn.Sequential(
        torch.nn.Linear(1024, 2048),
        torch.nn.BatchNorm1d(2048),
        # torch.nn.ReLU(),  # jia(11.8)
        # torch.nn.Dropout(p=0.5),           #try 0.75
        torch.nn.Linear(2048, 1502))      #1502

    self.classifer2 = torch.nn.Sequential(
        torch.nn.Linear(1024, 2048),
        torch.nn.BatchNorm1d(2048),
        # torch.nn.ReLU(),  # jia(11.8)
        # torch.nn.Dropout(p=0.5),           #try 0.75
        torch.nn.Linear(2048, 1502))

    self.classifer1_2 = torch.nn.Sequential(
        torch.nn.Linear(1024, 2048),         #3072
        torch.nn.BatchNorm1d(2048),
        # torch.nn.ReLU(),  # jia(11.8)
        # torch.nn.Dropout(p=0.5),           #try 0.75
        torch.nn.Linear(2048, 1502))

    self.classifer2_2 = torch.nn.Sequential(
        torch.nn.Linear(1024, 2048),
        torch.nn.BatchNorm1d(2048),
        # torch.nn.ReLU(),  # jia(11.8)
        # torch.nn.Dropout(p=0.5),           #try 0.75
        torch.nn.Linear(2048, 1502))

    self.classiferh1_2 = torch.nn.Sequential(
        torch.nn.Linear(1024, 2048),
        torch.nn.BatchNorm1d(2048),
        # torch.nn.ReLU(),  # jia(11.8)
        # torch.nn.Dropout(p=0.5),           #try 0.75
        torch.nn.Linear(2048, 1502))
    '''
    self.classifer3 = torch.nn.Sequential(
        torch.nn.Linear(2048, 6144),
        torch.nn.BatchNorm1d(6144),
        # torch.nn.ReLU(),  # jia(11.8)
        # torch.nn.Dropout(p=0.5),           #try 0.75
        torch.nn.Linear(6144, 1502))
    
    self.classifer4 = torch.nn.Sequential(
        torch.nn.Linear(2048, 6144),
        torch.nn.BatchNorm1d(6144),
        # torch.nn.ReLU(),  # jia(11.8)
        # torch.nn.Dropout(p=0.5),           #try 0.75
        torch.nn.Linear(6144, 1502))

    self.classifer5 = torch.nn.Sequential(
        torch.nn.Linear(4096, 6144),
        torch.nn.BatchNorm1d(6144),
        # torch.nn.ReLU(),  # jia(11.8)
        # torch.nn.Dropout(p=0.5),           #try 0.75
        torch.nn.Linear(6144, 1502))

    self.classifer6 = torch.nn.Sequential(
        torch.nn.Linear(2048, 6144),
        torch.nn.BatchNorm1d(6144),
        # torch.nn.ReLU(),  # jia(11.8)
        # torch.nn.Dropout(p=0.5),           #try 0.75
        torch.nn.Linear(6144, 1502))
    '''

    '''
    self.at1_x1 = torch.nn.Sequential(
        torch.nn.Conv2d(2048, 1, kernel_size=1, stride=1),  # False
        # torch.nn.BatchNorm2d(1024),
        # torch.nn.ReLU(),
        # torch.nn.Dropout(0.5),
        # torch.nn.Conv2d(1024, 1, kernel_size=1, stride=1),
        torch.nn.Sigmoid())

    self.at2_x1 = torch.nn.Sequential(
        torch.nn.Conv2d(2048, 1, kernel_size=1, stride=1),  # False
        # torch.nn.BatchNorm2d(1024),
        # torch.nn.ReLU(),
        # torch.nn.Dropout(0.5),
        # torch.nn.Conv2d(1024, 1, kernel_size=1, stride=1),
        torch.nn.Sigmoid())
    '''

    '''
    self.mid_dense1_x2 = torch.nn.Sequential(
        torch.nn.Linear(2048, 1024),
        torch.nn.BatchNorm1d(1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 128))
    '''
    #self.at2_x1_1 = torch.nn.Sequential(
    #    torch.nn.Conv2d(512, 1, kernel_size=1, stride=1))

    #self.at2_x1_2 = torch.nn.Sequential(
    #    torch.nn.Conv2d(512, 1, kernel_size=1, stride=1))

    #self.at2_x1_3 = torch.nn.Sequential(
     #   torch.nn.Conv2d(512, 1, kernel_size=1, stride=1))

    #self.at2_x1_4 = torch.nn.Sequential(
    #    torch.nn.Conv2d(512, 1, kernel_size=1, stride=1))
    '''
    self.spa_att = torch.nn.Sequential(
        torch.nn.Linear(192, 192),
        # torch.nn.BatchNorm1d(800),
        #torch.nn.ReLU(),
        #torch.nn.Dropout(0.5),  # try 0.75
        #torch.nn.Linear(576, 192),
        torch.nn.Sigmoid())
    '''
    #self.den1 = torch.nn.Linear(192, 1)
    #self.den = torch.nn.ModuleList([torch.nn.Linear(192, 1) for i in range(192)])
    #self.den2 = torch.nn.ModuleList([torch.nn.Linear(192, 1) for i in range(192)])
    #self.den3 = torch.nn.ModuleList([torch.nn.Linear(192, 1) for i in range(192)])
    #self.den4 = torch.nn.ModuleList([torch.nn.Linear(192, 1) for i in range(192)])
    #self.den = torch.nn.Linear(192, 1)
    #self.spa_map = torch.nn.ModuleList([torch.nn.Conv2d(2048, 1, kernel_size=1, stride=1) for i in range(192)])
    self.sig = torch.nn.Sigmoid()

    '''
    self.l_conv0 = nn.Conv2d(2048, 1, kernel_size=1, stride=1)
    self.bn0 = torch.nn.BatchNorm1d(2048)
    self.l_conv0_1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
    self.bn0_1 = nn.BatchNorm2d(1)

    self.l_conv1 = nn.Conv2d(2048, 1, kernel_size=1, stride=1)


    self.l_conv2 = torch.nn.Sequential(
        torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, bias=False,padding=1),  # False
        # torch.nn.BatchNorm2d(1024),
        #torch.nn.ReLU(),
        #torch.nn.Conv2d(1024, 1, kernel_size=1, stride=1, bias=False),
        torch.nn.Sigmoid())

    self.l_conv3 = torch.nn.Sequential(
        torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, bias=False, padding=1),  # False
        # torch.nn.BatchNorm2d(1024),
        # torch.nn.ReLU(),
        # torch.nn.Conv2d(1024, 1, kernel_size=1, stride=1, bias=False),
        torch.nn.Sigmoid())
    '''
    '''
    self.l_conv2 = torch.nn.Sequential(
        torch.nn.Conv2d(2048, 1024, kernel_size=1, stride=1, bias=False),  # False
        #torch.nn.BatchNorm2d(1024),
        torch.nn.ReLU(),
        torch.nn.Conv2d(1024, 1, kernel_size=1, stride=1, bias=False),
        torch.nn.Sigmoid())
    
    
    self.l_conv3 = torch.nn.Sequential(
        torch.nn.Conv2d(2048, 1024, kernel_size=1, stride=1, bias=False),  # False
        #torch.nn.BatchNorm2d(1024),
        torch.nn.ReLU(),
        torch.nn.Conv2d(1024, 1, kernel_size=1, stride=1, bias=False),
        torch.nn.Sigmoid())
    '''
    self.relu = nn.ReLU()
    '''
    self.l_conv4 = nn.Conv2d(2048, 1, kernel_size=1, stride=1)
    self.l_conv4_1 = nn.Conv2d(1, 1, kernel_size=(5,3), stride=1, padding=(2,1))
    self.bn4 = nn.BatchNorm2d(1)
    self.bn4_1 = nn.BatchNorm2d(1)

    self.l_conv5 = nn.Conv2d(128, 1, kernel_size=1, stride=1)

    self.bn1 = torch.nn.BatchNorm1d(2048)
    self.relu = nn.ReLU()

    self.bn2 = nn.BatchNorm2d(1)
    self.bn3 = nn.BatchNorm2d(1)

    self.bn4 = nn.BatchNorm2d(1)

    self.bn5 = nn.BatchNorm2d(1)

    self.bn_order = nn.BatchNorm2d(2048)
    '''
    '''
    self.l_0_conv1 = nn.Conv2d(2048, 1, kernel_size=1, stride=1)
    self.l_0_conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
    self.l_0_conv3 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
    self.bn_0_1 = nn.BatchNorm2d(1)
    self.bn_0_2 = nn.BatchNorm2d(1)
    self.bn_0_3 = nn.BatchNorm2d(1)
    '''

    '''
    self.dense_reduc1 = torch.nn.Sequential(
        torch.nn.Linear(2048, 1024),
        torch.nn.BatchNorm1d(1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 128))

    self.dense_reduc2 = torch.nn.Sequential(
        torch.nn.Linear(512, 1024),
        torch.nn.BatchNorm1d(1024),
        torch.nn.ReLU())

    self.dense_reduc3 = torch.nn.Sequential(
        torch.nn.Linear(512, 1024),
        torch.nn.BatchNorm1d(1024),
        torch.nn.ReLU())

    self.dense_reduc0 = torch.nn.Sequential(
        torch.nn.Linear(2048, 1024),
        torch.nn.BatchNorm1d(1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 128))

    self.dense_reduc4 = torch.nn.Sequential(
        torch.nn.Linear(512, 1024),
        torch.nn.BatchNorm1d(1024),
        torch.nn.ReLU())

    self.u_p_1 = torch.nn.Sequential(
        torch.nn.Conv2d(2048, 1024, kernel_size=1, stride=1, bias=False),  # False
        #torch.nn.BatchNorm2d(1024),
        torch.nn.ReLU())
        #torch.nn.Conv2d(1024, 1, kernel_size=1, stride=1, bias=False),
        #torch.nn.Sigmoid())

    self.u_p_2 = torch.nn.Sequential(
        torch.nn.Conv2d(2048, 1024, kernel_size=1, stride=1, bias=False),  # False
        #torch.nn.BatchNorm2d(1024),
        torch.nn.ReLU())
    # torch.nn.Conv2d(1024, 1, kernel_size=1, stride=1, bias=False),
    # torch.nn.Sigmoid())

    self.p = torch.nn.Sequential(
        torch.nn.Conv2d(1024, 1024, kernel_size=1, stride=1, bias=False,groups=1024))  # False
        #torch.nn.BatchNorm2d(1024))
        #torch.nn.ReLU())

    self.u_1 = torch.nn.Sequential(
        torch.nn.Conv2d(2048, 1024, kernel_size=1, stride=1, bias=False),  # False
        #torch.nn.BatchNorm2d(1024),
        torch.nn.ReLU())

    self.u_2 = torch.nn.Sequential(
        torch.nn.Conv2d(2048, 1024, kernel_size=1, stride=1, bias=False),  # False
        #torch.nn.BatchNorm2d(1024),
        torch.nn.ReLU())
    '''
    #self.conv5a = nn.Sequential(nn.Conv2d(2048, 512, 3, padding=1, bias=False),
    #                            nn.BatchNorm2d(512),
    #                            nn.ReLU())
    self.conv5a = nn.Sequential(nn.Conv2d(1024, 512, 1, bias=False),
                                nn.BatchNorm2d(512),
                                nn.ReLU())

    self.conv5b = nn.Sequential(nn.Conv2d(2048, 512, 1, bias=False),
                                nn.BatchNorm2d(512),
                                nn.ReLU())


    self.query_conv = Conv2d(in_channels=512, out_channels=512 // 16, kernel_size=1)
    self.key_conv = Conv2d(in_channels=512, out_channels=512 // 16, kernel_size=1)

    self.value_conv = Conv2d(in_channels=512, out_channels=512, kernel_size=1)
    self.value_conv1 = Conv2d(in_channels=512, out_channels=512, kernel_size=1)

    #self.gamma = Parameter(torch.zeros(1))
    p0 = torch.ones(1) * 0.5
    self.gamma = Parameter(p0)

    self.ggg = Parameter(torch.zeros(1))

    self.softmax = Softmax(dim=-1)

    #self.conv51 = nn.Sequential(nn.Conv2d(512, 2048, 3, padding=1, bias=False),
    #                            nn.BatchNorm2d(2048),
    #                            nn.ReLU())
    self.conv51 = nn.Sequential(nn.Conv2d(512, 1024, 1, bias=False),
                                nn.BatchNorm2d(1024),
                                nn.ReLU())

    #self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, 2048, 1))

    ## 2 ##
    self.s_query_conv = Conv2d(in_channels=512, out_channels=512 // 16, kernel_size=1)
    self.s_key_conv = Conv2d(in_channels=512, out_channels=512 // 16, kernel_size=1)


    self.s_value_conv = Conv2d(in_channels=512, out_channels=512, kernel_size=1)
    self.s_value_conv1 = Conv2d(in_channels=512, out_channels=512, kernel_size=1)

    #self.gamma1 = Parameter(torch.zeros(1))
    p1 = torch.ones(1)*0.5
    self.gamma1 = Parameter(p1)

    self.conv52 = nn.Sequential(nn.Conv2d(512, 1024, 1, bias=False),
                                nn.BatchNorm2d(1024),
                                nn.ReLU())

    self.mask1 =  torch.nn.Sequential(
                  torch.nn.Conv2d(512, 1, kernel_size=1, stride=1),  # False
                  torch.nn.BatchNorm2d(1),
                  # torch.nn.ReLU(),
                  # torch.nn.Dropout(0.5),
                  # torch.nn.Conv2d(1024, 1, kernel_size=1, stride=1),
                  torch.nn.Sigmoid())

    self.mask2 = torch.nn.Sequential(
                  torch.nn.Conv2d(512, 1, kernel_size=1, stride=1),  # False
                  torch.nn.BatchNorm2d(1),
                  # torch.nn.ReLU(),
                  # torch.nn.Dropout(0.5),
                  # torch.nn.Conv2d(1024, 1, kernel_size=1, stride=1),
                  torch.nn.Sigmoid())
    '''
    self.dense_y3 = torch.nn.Sequential(
        torch.nn.Linear(2048, 2048),
        torch.nn.BatchNorm1d(2048),
        torch.nn.ReLU())
    '''
    self.dense_y1 = torch.nn.Sequential(
        torch.nn.Linear(1024, 1024),
        torch.nn.BatchNorm1d(1024),
        torch.nn.ReLU())
        #torch.nn.Linear(1024, 128))

    self.dense_y2 = torch.nn.Sequential(
        torch.nn.Linear(1024, 1024),
        torch.nn.BatchNorm1d(1024),
        torch.nn.ReLU())
        #torch.nn.Linear(1024, 128))



    ### 2 and 4 ###
    self.conv6a = nn.Sequential(nn.Conv2d(1024, 512, 1, bias=False),
                                nn.BatchNorm2d(512),
                                nn.ReLU())

    self.conv6b = nn.Sequential(nn.Conv2d(1024, 512, 1, bias=False),
                                nn.BatchNorm2d(512),
                                nn.ReLU())

    self.mask_2 = torch.nn.Sequential(
        torch.nn.Conv2d(512, 1, kernel_size=1, stride=1),  # False
        torch.nn.BatchNorm2d(1),
        # torch.nn.ReLU(),
        # torch.nn.Dropout(0.5),
        # torch.nn.Conv2d(1024, 1, kernel_size=1, stride=1),
        torch.nn.Sigmoid())

    self.mask_4 = torch.nn.Sequential(
        torch.nn.Conv2d(512, 1, kernel_size=1, stride=1),  # False
        torch.nn.BatchNorm2d(1),
        # torch.nn.ReLU(),
        # torch.nn.Dropout(0.5),
        # torch.nn.Conv2d(1024, 1, kernel_size=1, stride=1),
        torch.nn.Sigmoid())

    self.query_conv_2 = Conv2d(in_channels=512, out_channels=512 // 16, kernel_size=1)
    self.key_conv_2 = Conv2d(in_channels=512, out_channels=512 // 16, kernel_size=1)

    self.value_conv_2= Conv2d(in_channels=512, out_channels=512, kernel_size=1)
    self.value_conv1_2 = Conv2d(in_channels=512, out_channels=512, kernel_size=1)

    self.s_query_conv_2 = Conv2d(in_channels=512, out_channels=512 // 16, kernel_size=1)
    self.s_key_conv_2 = Conv2d(in_channels=512, out_channels=512 // 16, kernel_size=1)

    self.s_value_conv_2 = Conv2d(in_channels=512, out_channels=512, kernel_size=1)
    self.s_value_conv1_2 = Conv2d(in_channels=512, out_channels=512, kernel_size=1)

    self.conv61 = nn.Sequential(nn.Conv2d(512, 1024, 1, bias=False),
                                nn.BatchNorm2d(1024),
                                nn.ReLU())

    self.conv62 = nn.Sequential(nn.Conv2d(512, 1024, 1, bias=False),
                                nn.BatchNorm2d(1024),
                                nn.ReLU())

    #self.x2_conv = nn.Sequential(nn.Conv2d(2048, 1024, 1, bias=False),
    #                            nn.BatchNorm2d(1024),
     #                           nn.ReLU())

    self.dense_y1_2 = torch.nn.Sequential(
        torch.nn.Linear(1024, 1024),
        torch.nn.BatchNorm1d(1024),
        torch.nn.ReLU())
    # torch.nn.Linear(1024, 128))

    self.dense_y2_2 = torch.nn.Sequential(
        torch.nn.Linear(1024, 1024),
        torch.nn.BatchNorm1d(1024),
        torch.nn.ReLU())

    self.dense_h1_2 = torch.nn.Sequential(
        torch.nn.Linear(1024, 1024),
        torch.nn.BatchNorm1d(1024),
        torch.nn.ReLU())
    '''
    self.dense_reduc_0_1 = torch.nn.Sequential(
        torch.nn.Linear(2048, 1024),
        torch.nn.BatchNorm1d(1024),
        torch.nn.ReLU())

    self.dense_reduc_0_2 = torch.nn.Sequential(
        torch.nn.Linear(2048, 1024),
        torch.nn.BatchNorm1d(1024),
        torch.nn.ReLU())
    '''

    '''
    self.conv2 = torch.nn.Sequential(
        nn.Conv2d(2048, 2048, kernel_size=3, stride=1,padding=1, bias=False),
        torch.nn.Sigmoid())
    '''
    '''
    self.dense1_x1 = torch.nn.Sequential(
        torch.nn.Linear(2048, 1024),
        torch.nn.BatchNorm1d(1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 128))
    '''
  def forward(self, x,label):

    # shape [N, C, H, W]  [128, 2048, 16, 8]
    #x = self.base(x)
    x = self.backone(x)

    x00 = self.p00(x)  #3_1
    x0 = self.p0(x00)  #3
    x1 = self.p1(x0)      #[128, 2048, 24, 8]
    #x1 = self.part_2(x1)
    #x2 = self.p2(x)      #[128, 2048, 24, 8]


    feat1 = self.conv5a(x0)
    feat2 = self.conv5b(x1)

    feat1_mask = self.mask1(feat1)
    feat1 = torch.mul(feat1_mask, feat1)

    feat2_mask = self.mask2(feat2)
    feat2 = torch.mul(feat2_mask, feat2)

    m_batchsize, C, height, width = feat1.size()

    proj_query = self.query_conv(feat1).view(m_batchsize, -1, width * height).permute(0, 2, 1)
    proj_query = self.relu(proj_query)

    proj_key = self.key_conv(feat2).view(m_batchsize, -1, width * height)
    proj_key = self.relu(proj_key)
    energy = torch.bmm(proj_query, proj_key)

    attention = self.softmax(energy)
    proj_value = self.value_conv(feat1)
    proj_value1 = self.value_conv1(feat2).view(m_batchsize, -1, width * height)

    out = torch.bmm(proj_value1, attention.permute(0, 2, 1))
    out = out.view(m_batchsize, C, height, width)
    #out = torch.einsum('bqk,bqk->bq',(out,proj_value1))/192
    out = torch.mul(out, proj_value)

    #out = out.view(m_batchsize, C, height, width)


    #out = self.gamma * out + feat1
    #out = feat1

    ## 2 ##
    s_proj_query = self.s_query_conv(feat1).view(m_batchsize, -1, width * height)
    s_proj_query = self.relu(s_proj_query)

    s_proj_key = self.s_key_conv(feat2).view(m_batchsize, -1, width * height).permute(0, 2, 1)
    s_proj_key = self.relu(s_proj_key)
    s_energy = torch.bmm(s_proj_key, s_proj_query)

    s_attention = self.softmax(s_energy)
    s_proj_value = self.s_value_conv(feat1).view(m_batchsize, -1, width * height)
    s_proj_value1 = self.s_value_conv1(feat2)

    s_out = torch.bmm(s_proj_value, s_attention.permute(0, 2, 1))
    s_out = s_out.view(m_batchsize, C, height, width)
    # out = torch.einsum('bqk,bqk->bq',(out,proj_value1))/192
    s_out = torch.mul(s_out, s_proj_value1)

    # out = out.view(m_batchsize, C, height, width)

    #s_out = self.gamma1 * s_out + feat2
    #s_out = feat2

    out = self.conv51(out)
    out1 = self.conv52(s_out)






    mid_zeros_1 = torch.zeros(out.size(0), 1024, 192)
    mid_zeros_1 = mid_zeros_1.cuda()

    mid_ones_1 = torch.ones(out.size(0), 1024, 192)
    mid_ones_1 = mid_ones_1.cuda()

    mid_zeros_0 = mid_zeros_1
    mid_ones_0 = mid_ones_1

    '''
    mid_zeros_0 = torch.zeros(out.size(0), 2048, 192)
    mid_zeros_0 = mid_zeros_0.cuda()

    mid_ones_0 = torch.ones(out.size(0), 2048, 192)
    mid_ones_0 = mid_ones_0.cuda()
    '''
    #
    out_1 = out.view(out.size(0),1024,192)
    out1_min = out_1.min(2)[0].unsqueeze(dim=2).repeat(1, 1, 192)
    out1_inter = out_1 - out1_min
    out1_max = out1_inter.max(2)[0].unsqueeze(dim=2).repeat(1, 1, 192) + 1e-10
    out1_inter = out1_inter/out1_max

    p_yu1 = out1_inter.max(2)[0]
    p_tt = p_yu1 * 0.7
    p_cc = p_yu1 * 0.3
    p_yu1 = p_yu1 * 0.5
    p_yu1 = p_yu1.unsqueeze(dim=2)
    p_yu1 = p_yu1.repeat(1, 1, 192) - 1e-10

    mid_erase1 = torch.where(out1_inter > p_yu1, mid_ones_0, mid_zeros_0)

    num1 = torch.sum(mid_erase1, 2)
    #print (num1)
    #num1 = 1 / num1

    #num1 = num1.unsqueeze(dim=1).unsqueeze(dim=1)

    y1_1_m1 = torch.mul(out_1, mid_erase1)
    y1_1_m1 = y1_1_m1.sum(dim=2)
    #y1_1_m1 = torch.mul(y1_1_m1, num1)
    y1_1_m1 = y1_1_m1 / num1
    #
    '''
    p_tt = p_tt.unsqueeze(dim=2)
    p_tt = p_tt.repeat(1, 1, 192) - 1e-10

    mid_tt1 = torch.where(out1_inter > p_tt, mid_ones_0, mid_zeros_0)

    tt1 = torch.sum(mid_tt1, 2)

    y1_tt = torch.mul(out_1, mid_tt1)
    y1_tt = y1_tt.sum(dim=2)
    # y1_1_m1 = torch.mul(y1_1_m1, num1)
    y1_tt = y1_tt / tt1
    '''
    #

    p_cc = p_cc.unsqueeze(dim=2)
    p_cc = p_cc.repeat(1, 1, 192) - 1e-10

    mid_cc1 = torch.where(out1_inter > p_cc, mid_ones_0, mid_zeros_0)

    cc1 = torch.sum(mid_cc1, 2)

    y1_cc = torch.mul(out_1, mid_cc1)
    y1_cc = y1_cc.sum(dim=2)
    # y1_1_m1 = torch.mul(y1_1_m1, num1)
    y1_cc = y1_cc / cc1
    #


    y1_max = F.max_pool2d(out, out.size()[2:])
    y1_1 = F.avg_pool2d(out, out.size()[2:])
    y1_1 = y1_1 + y1_max
    y1_1 = y1_1.view(y1_1.size(0), -1)
    y1_1 = y1_1 + y1_1_m1 + y1_cc

    y1_1 = torch.sign(y1_1) * torch.sqrt(torch.abs(y1_1) + 1e-10)
    y1_1 = self.dense_y1(y1_1)
    #y1_1 = out

    #y1_2 = F.avg_pool2d(x1_2, x1_2.size()[2:])
    #y1_2 = y1_2.view(y1_2.size(0), -1)

    #y1_3 = F.avg_pool2d(x1_3, x1_3.size()[2:])
    #y1_3 = y1_3.view(y1_3.size(0), -1)


    y1_x1_score_v = self.classifer1(y1_1)
    y1_x1_score_v = F.log_softmax(y1_x1_score_v, 1)


    #
    out_2 = out1.view(out1.size(0), 1024, 192)
    out2_min = out_2.min(2)[0].unsqueeze(dim=2).repeat(1, 1, 192)
    out2_inter = out_2 - out2_min
    out2_max = out2_inter.max(2)[0].unsqueeze(dim=2).repeat(1, 1, 192) + 1e-10
    out2_inter = out2_inter / out2_max

    p_yu2 = out2_inter.max(2)[0]
    p_yy = p_yu2 * 0.7
    p_vv = p_yu2 * 0.3
    p_yu2 = p_yu2 * 0.5
    p_yu2 = p_yu2.unsqueeze(dim=2)
    p_yu2 = p_yu2.repeat(1, 1, 192) - 1e-10

    mid_erase2 = torch.where(out2_inter > p_yu2, mid_ones_1, mid_zeros_1)

    num2 = torch.sum(mid_erase2, 2)

    #print (num2)
    #num2 = 1 / num2

    # num1 = num1.unsqueeze(dim=1).unsqueeze(dim=1)

    y1_2_m1 = torch.mul(out_2, mid_erase2)
    y1_2_m1 = y1_2_m1.sum(dim=2)
    #y1_2_m1 = torch.mul(y1_2_m1, num2)
    y1_2_m1 = y1_2_m1/num2
    #print (num2)
    #
    '''
    p_yy = p_yy.unsqueeze(dim=2)
    p_yy = p_yy.repeat(1, 1, 192) - 1e-10

    mid_yy = torch.where(out2_inter > p_yy, mid_ones_1, mid_zeros_1)

    yy2 = torch.sum(mid_yy, 2)

    # print (num2)
    # num2 = 1 / num2

    # num1 = num1.unsqueeze(dim=1).unsqueeze(dim=1)

    y1_yy = torch.mul(out_2, mid_yy)
    y1_yy = y1_yy.sum(dim=2)
    # y1_2_m1 = torch.mul(y1_2_m1, num2)
    y1_yy = y1_yy / yy2
    '''
    #

    p_vv = p_vv.unsqueeze(dim=2)
    p_vv = p_vv.repeat(1, 1, 192) - 1e-10

    mid_vv = torch.where(out2_inter > p_vv, mid_ones_1, mid_zeros_1)

    vv2 = torch.sum(mid_vv, 2)

    # print (num2)
    # num2 = 1 / num2

    # num1 = num1.unsqueeze(dim=1).unsqueeze(dim=1)

    y1_vv = torch.mul(out_2, mid_vv)
    y1_vv = y1_vv.sum(dim=2)
    # y1_2_m1 = torch.mul(y1_2_m1, num2)
    y1_vv = y1_vv / vv2
    #




    y2_max = F.max_pool2d(out1, out1.size()[2:])
    y1_2 = F.avg_pool2d(out1, out1.size()[2:])
    y1_2 = y1_2 + y2_max
    y1_2 = y1_2.view(y1_2.size(0), -1)
    y1_2 = y1_2 + y1_2_m1 + y1_vv

    y1_2 = torch.sign(y1_2) * torch.sqrt(torch.abs(y1_2) + 1e-10)
    y1_2 = self.dense_y2(y1_2)

    y2_x1_score_v = self.classifer2(y1_2)
    y2_x1_score_v = F.log_softmax(y2_x1_score_v, 1)


    #mid_zeros_1 = torch.zeros(x1.size(0), 2048, 192)
    #mid_zeros_1 = mid_zeros_1.cuda()

    #mid_ones_1 = torch.ones(x1.size(0), 2048, 192)
    #mid_ones_1 = mid_ones_1.cuda()

    #m_batchsize, C, height, width = x1.size(0), 512, 24, 8


                ###  2 and 4 ###
    '''
    mid_zeros_1_2 = torch.zeros(x1.size(0), 2048, 192)
    mid_zeros_1_2 = mid_zeros_1_2.cuda()

    mid_ones_1_2 = torch.ones(x1.size(0), 2048, 192)
    mid_ones_1_2 = mid_ones_1_2.cuda()
    '''
    mid_zeros_1_p = torch.zeros(out.size(0), 1024, 192)
    mid_zeros_1_p = mid_zeros_1_p.cuda()

    mid_ones_1_p = torch.ones(out.size(0), 1024, 192)
    mid_ones_1_p = mid_ones_1_p.cuda()

    mid_zeros_0_p = mid_zeros_1_p
    mid_ones_0_p = mid_ones_1_p



    fea_2 = self.conv6a(x00)
    fea_4 = self.conv6b(x0)

    fea_2_mask = self.mask_2(fea_2)
    fea_2 = torch.mul(fea_2_mask, fea_2)

    fea_4_mask = self.mask_4(fea_4)
    fea_4 = torch.mul(fea_4_mask, fea_4)

    m_batchsize_2, C_2, height_2, width_2 = fea_2.size()

    proj_query_2 = self.query_conv_2(fea_2).view(m_batchsize_2, -1, width_2 * height_2).permute(0, 2, 1)
    proj_query_2 = self.relu(proj_query_2)

    proj_key_4 = self.key_conv_2(fea_4).view(m_batchsize, -1, width * height)
    proj_key_4 = self.relu(proj_key_4)
    energy_2 = torch.bmm(proj_query_2, proj_key_4)

    attention_2 = self.softmax(energy_2)
    proj_value_2 = self.value_conv_2(fea_2)
    proj_value1_4 = self.value_conv1_2(fea_4).view(m_batchsize, -1, width * height)

    out_2 = torch.bmm(proj_value1_4, attention_2.permute(0, 2, 1))
    out_2 = out_2.view(m_batchsize_2, C_2, height_2, width_2)
    # out = torch.einsum('bqk,bqk->bq',(out,proj_value1))/192
    out_2 = torch.mul(out_2, proj_value_2)

    # out = out.view(m_batchsize, C, height, width)

    # out = self.gamma * out + feat1
    # out = feat1

    ## 2 ##
    s_proj_query_2 = self.s_query_conv_2(fea_2).view(m_batchsize_2, -1, height_2 * width_2)
    s_proj_query_2 = self.relu(s_proj_query_2)

    s_proj_key_2 = self.s_key_conv_2(fea_4).view(m_batchsize, -1, width * height).permute(0, 2, 1)
    s_proj_key_2 = self.relu(s_proj_key_2)
    s_energy_2 = torch.bmm(s_proj_key_2, s_proj_query_2)

    s_attention_2 = self.softmax(s_energy_2)
    s_proj_value_2 = self.s_value_conv_2(fea_2).view(m_batchsize_2, -1, height_2 * width_2)
    s_proj_value1_2 = self.s_value_conv1_2(fea_4)

    s_out_2 = torch.bmm(s_proj_value_2, s_attention_2.permute(0, 2, 1))
    s_out_2 = s_out_2.view(m_batchsize, C, height, width)
    # out = torch.einsum('bqk,bqk->bq',(out,proj_value1))/192
    s_out_2 = torch.mul(s_out_2, s_proj_value1_2)

    # out = out.view(m_batchsize, C, height, width)

    # s_out = self.gamma1 * s_out + feat2
    # s_out = feat2

    out_2 = self.conv61(out_2)
    out1_2 = self.conv62(s_out_2)


    #
    out_1_2 = out_2.view(out_2.size(0), 1024, 192)
    out1_min_2 = out_1_2.min(2)[0].unsqueeze(dim=2).repeat(1, 1, 192)
    out1_inter_2 = out_1_2 - out1_min_2
    out1_max_2 = out1_inter_2.max(2)[0].unsqueeze(dim=2).repeat(1, 1, 192) + 1e-10
    out1_inter_2 = out1_inter_2 / out1_max_2

    p_yu1_2 = out1_inter_2.max(2)[0]
    p_uu = p_yu1_2 * 0.7
    p_bb = p_yu1_2 * 0.3
    p_yu1_2 = p_yu1_2 * 0.5
    p_yu1_2 = p_yu1_2.unsqueeze(dim=2)
    p_yu1_2 = p_yu1_2.repeat(1, 1, 192) - 1e-10

    mid_erase1_2 = torch.where(out1_inter_2 > p_yu1_2, mid_ones_0_p, mid_zeros_0_p)

    num1_2 = torch.sum(mid_erase1_2, 2)
    # print (num1)
    # num1 = 1 / num1

    # num1 = num1.unsqueeze(dim=1).unsqueeze(dim=1)

    y1_1_m1_2 = torch.mul(out_1_2, mid_erase1_2)
    y1_1_m1_2 = y1_1_m1_2.sum(dim=2)
    # y1_1_m1 = torch.mul(y1_1_m1, num1)
    y1_1_m1_2 = y1_1_m1_2 / num1_2

    #
    '''
    p_uu = p_uu.unsqueeze(dim=2)
    p_uu = p_uu.repeat(1, 1, 192) - 1e-10

    mid_uu = torch.where(out1_inter_2 > p_uu, mid_ones_0, mid_zeros_0)

    uu2 = torch.sum(mid_uu, 2)
    # print (num1)
    # num1 = 1 / num1

    # num1 = num1.unsqueeze(dim=1).unsqueeze(dim=1)

    y1_uu = torch.mul(out_1_2, mid_uu)
    y1_uu = y1_uu.sum(dim=2)
    # y1_1_m1 = torch.mul(y1_1_m1, num1)
    y1_uu = y1_uu / uu2
    '''
    #


    p_bb = p_bb.unsqueeze(dim=2)
    p_bb = p_bb.repeat(1, 1, 192) - 1e-10

    mid_bb = torch.where(out1_inter_2 > p_bb, mid_ones_0_p, mid_zeros_0_p)

    bb2 = torch.sum(mid_bb, 2)
    # print (num1)
    # num1 = 1 / num1

    # num1 = num1.unsqueeze(dim=1).unsqueeze(dim=1)

    y1_bb = torch.mul(out_1_2, mid_bb)
    y1_bb = y1_bb.sum(dim=2)
    # y1_1_m1 = torch.mul(y1_1_m1, num1)
    y1_bb = y1_bb / bb2
    #


    y1_max_2 = F.max_pool2d(out_2, out_2.size()[2:])
    y1_1_2 = F.avg_pool2d(out_2, out_2.size()[2:])
    y1_1_2 = y1_1_2 + y1_max_2
    y1_1_2 = y1_1_2.view(y1_1_2.size(0), -1)
    y1_1_2 = y1_1_2 + y1_1_m1_2 + y1_bb

    y1_1_2 = torch.sign(y1_1_2) * torch.sqrt(torch.abs(y1_1_2) + 1e-10)
    y1_1_2 = self.dense_y1_2(y1_1_2)
    # y1_1 = out

    # y1_2 = F.avg_pool2d(x1_2, x1_2.size()[2:])
    # y1_2 = y1_2.view(y1_2.size(0), -1)

    # y1_3 = F.avg_pool2d(x1_3, x1_3.size()[2:])
    # y1_3 = y1_3.view(y1_3.size(0), -1)

    y1_x1_score_v_2 = self.classifer1_2(y1_1_2)
    y1_x1_score_v_2 = F.log_softmax(y1_x1_score_v_2, 1)





    #
    out_2_2 = out1_2.view(out1_2.size(0), 1024, 192)
    out2_min_2 = out_2_2.min(2)[0].unsqueeze(dim=2).repeat(1, 1, 192)
    out2_inter_2 = out_2_2 - out2_min_2
    out2_max_2 = out2_inter_2.max(2)[0].unsqueeze(dim=2).repeat(1, 1, 192) + 1e-10
    out2_inter_2 = out2_inter_2 / out2_max_2

    p_yu2_2 = out2_inter_2.max(2)[0]
    p_ii = p_yu2_2 * 0.7
    p_nn = p_yu2_2 * 0.3
    p_yu2_2 = p_yu2_2 * 0.5
    p_yu2_2 = p_yu2_2.unsqueeze(dim=2)
    p_yu2_2 = p_yu2_2.repeat(1, 1, 192) - 1e-10

    mid_erase2_2 = torch.where(out2_inter_2 > p_yu2_2, mid_ones_0_p, mid_zeros_0_p)

    num2_2 = torch.sum(mid_erase2_2, 2)

    # print (num2)
    # num2 = 1 / num2

    # num1 = num1.unsqueeze(dim=1).unsqueeze(dim=1)

    y1_2_m1_2 = torch.mul(out_2_2, mid_erase2_2)
    y1_2_m1_2 = y1_2_m1_2.sum(dim=2)
    # y1_2_m1 = torch.mul(y1_2_m1, num2)
    y1_2_m1_2 = y1_2_m1_2 / num2_2
    # print (num2)
    #

    p_nn = p_nn.unsqueeze(dim=2)
    p_nn = p_nn.repeat(1, 1, 192) - 1e-10

    mid_nn = torch.where(out2_inter_2 > p_nn, mid_ones_0_p, mid_zeros_0_p)

    nn2 = torch.sum(mid_nn, 2)

    # print (num2)
    # num2 = 1 / num2

    # num1 = num1.unsqueeze(dim=1).unsqueeze(dim=1)

    y1_nn = torch.mul(out_2_2, mid_nn)
    y1_nn = y1_nn.sum(dim=2)
    # y1_2_m1 = torch.mul(y1_2_m1, num2)
    y1_nn = y1_nn / nn2
    #
    '''
    p_ii = p_ii.unsqueeze(dim=2)
    p_ii = p_ii.repeat(1, 1, 192) - 1e-10

    mid_ii = torch.where(out2_inter_2 > p_ii, mid_ones_0, mid_zeros_0)

    ii2 = torch.sum(mid_ii, 2)

    # print (num2)
    # num2 = 1 / num2

    # num1 = num1.unsqueeze(dim=1).unsqueeze(dim=1)

    y1_ii = torch.mul(out_2_2, mid_ii)
    y1_ii = y1_ii.sum(dim=2)
    # y1_2_m1 = torch.mul(y1_2_m1, num2)
    y1_ii = y1_ii / ii2
    '''
    #

    
    y2_max_2 = F.max_pool2d(out1_2, out1_2.size()[2:])
    y1_2_2 = F.avg_pool2d(out1_2, out1_2.size()[2:])
    y1_2_2 = y1_2_2 + y2_max_2
    y1_2_2 = y1_2_2.view(y1_2_2.size(0), -1)
    y1_2_2 = y1_2_2 + y1_2_m1_2 + y1_nn

    y1_2_2 = torch.sign(y1_2_2) * torch.sqrt(torch.abs(y1_2_2) + 1e-10)
    y1_2_2 = self.dense_y2_2(y1_2_2)

    y2_x1_score_v_2 = self.classifer2_2(y1_2_2)
    y2_x1_score_v_2 = F.log_softmax(y2_x1_score_v_2, 1)


    ### all ###


    #x2_input = self.x2_conv(out_2)
    #x2_input = self.ggg*x2_input + x00
    x2_input = x00
    h0 = self.Y0(x2_input)
    #h0 = h0 + out1_2 + outf
    h11 = self.Y1(h0)




    ###
    feat1 = self.conv5a(h0)
    feat2 = self.conv5b(h11)

    feat1_mask = self.mask1(feat1)
    feat1 = torch.mul(feat1_mask, feat1)

    feat2_mask = self.mask2(feat2)
    feat2 = torch.mul(feat2_mask, feat2)

    m_batchsize, C, height, width = feat1.size()

    s_proj_query = self.s_query_conv(feat1).view(m_batchsize, -1, width * height)
    s_proj_query = self.relu(s_proj_query)

    s_proj_key = self.s_key_conv(feat2).view(m_batchsize, -1, width * height).permute(0, 2, 1)
    s_proj_key = self.relu(s_proj_key)
    s_energy = torch.bmm(s_proj_key, s_proj_query)

    s_attention = self.softmax(s_energy)
    s_proj_value = self.s_value_conv(feat1).view(m_batchsize, -1, width * height)
    s_proj_value1 = self.s_value_conv1(feat2)

    s_out = torch.bmm(s_proj_value, s_attention.permute(0, 2, 1))
    s_out = s_out.view(m_batchsize, C, height, width)
    # out = torch.einsum('bqk,bqk->bq',(out,proj_value1))/192
    s_out = torch.mul(s_out, s_proj_value1)

    p2_bbb = self.conv52(s_out)

    ## global ##
    mid_zeros = torch.zeros(p2_bbb.size(0), 1024, 24, 8)
    mid_zeros = mid_zeros.cuda()

    ## 1 ##
    p2_all_max = F.max_pool2d(p2_bbb, p2_bbb.size()[2:])
    p2_all_avg = F.avg_pool2d(p2_bbb, p2_bbb.size()[2:])
    p2_all_avg = p2_all_max + p2_all_avg

    p2_parts_group = [None] * 24

    ppp_base = p2_bbb + 0

    p2_parts_group[0] = ppp_base[:, :, 1:23, :]
    local_feat_avg = F.max_pool2d(p2_parts_group[0], p2_parts_group[0].size()[2:])
    local_feat_avg_kuo = local_feat_avg.repeat(1, 1, 24, 8)
    ppp_base1 = torch.where(ppp_base == local_feat_avg_kuo, mid_zeros, ppp_base)

    p2_parts_group[1] = ppp_base1[:, :, 2:24, :]
    local_feat_avg = F.max_pool2d(p2_parts_group[1], p2_parts_group[1].size()[2:])
    local_feat_avg_kuo = local_feat_avg.repeat(1, 1, 24, 8)
    ppp_base2 = torch.where(ppp_base1 == local_feat_avg_kuo, mid_zeros, ppp_base1)

    p2_parts_group[2] = ppp_base2[:, :, 0:20, :]
    local_feat_avg = F.max_pool2d(p2_parts_group[2], p2_parts_group[2].size()[2:])
    local_feat_avg_kuo = local_feat_avg.repeat(1, 1, 24, 8)
    ppp_base3 = torch.where(ppp_base2 == local_feat_avg_kuo, mid_zeros, ppp_base2)

    p2_parts_group[3] = ppp_base3[:, :, 3:21, :]
    local_feat_avg = F.max_pool2d(p2_parts_group[3], p2_parts_group[3].size()[2:])
    local_feat_avg_kuo = local_feat_avg.repeat(1, 1, 24, 8)
    ppp_base4 = torch.where(ppp_base3 == local_feat_avg_kuo, mid_zeros, ppp_base3)

    p2_parts_group[4] = ppp_base4[:, :, 0:18, :]
    local_feat_avg = F.max_pool2d(p2_parts_group[4], p2_parts_group[4].size()[2:])
    local_feat_avg_kuo = local_feat_avg.repeat(1, 1, 24, 8)
    ppp_base5 = torch.where(ppp_base4 == local_feat_avg_kuo, mid_zeros, ppp_base4)

    p2_parts_group[5] = ppp_base5[:, :, 2:19, :]
    local_feat_avg = F.max_pool2d(p2_parts_group[5], p2_parts_group[5].size()[2:])
    local_feat_avg_kuo = local_feat_avg.repeat(1, 1, 24, 8)
    ppp_base6 = torch.where(ppp_base5 == local_feat_avg_kuo, mid_zeros, ppp_base5)

    p2_parts_group[6] = ppp_base6[:, :, 3:20, :]
    local_feat_avg = F.max_pool2d(p2_parts_group[6], p2_parts_group[6].size()[2:])
    local_feat_avg_kuo = local_feat_avg.repeat(1, 1, 24, 8)
    ppp_base7 = torch.where(ppp_base6 == local_feat_avg_kuo, mid_zeros, ppp_base6)

    p2_parts_group[7] = ppp_base7[:, :, 4:21, :]
    local_feat_avg = F.max_pool2d(p2_parts_group[7], p2_parts_group[7].size()[2:])
    local_feat_avg_kuo = local_feat_avg.repeat(1, 1, 24, 8)
    ppp_base8 = torch.where(ppp_base7 == local_feat_avg_kuo, mid_zeros, ppp_base7)

    p2_parts_group[8] = ppp_base8[:, :, 4:19, :]
    local_feat_avg = F.max_pool2d(p2_parts_group[8], p2_parts_group[8].size()[2:])
    local_feat_avg_kuo = local_feat_avg.repeat(1, 1, 24, 8)
    ppp_base9 = torch.where(ppp_base8 == local_feat_avg_kuo, mid_zeros, ppp_base8)

    p2_parts_group[9] = ppp_base9[:, :, 4:18, :]
    local_feat_avg = F.max_pool2d(p2_parts_group[9], p2_parts_group[9].size()[2:])
    local_feat_avg_kuo = local_feat_avg.repeat(1, 1, 24, 8)
    ppp_base10 = torch.where(ppp_base9 == local_feat_avg_kuo, mid_zeros, ppp_base9)

    p2_parts_group[10] = ppp_base10[:, :, 10:24, :]
    local_feat_avg = F.max_pool2d(p2_parts_group[10], p2_parts_group[10].size()[2:])
    local_feat_avg_kuo = local_feat_avg.repeat(1, 1, 24, 8)
    ppp_base11 = torch.where(ppp_base10 == local_feat_avg_kuo, mid_zeros, ppp_base10)

    p2_parts_group[11] = ppp_base11[:, :, 11:24, :]
    local_feat_avg = F.max_pool2d(p2_parts_group[11], p2_parts_group[11].size()[2:])
    local_feat_avg_kuo = local_feat_avg.repeat(1, 1, 24, 8)
    ppp_base12 = torch.where(ppp_base11 == local_feat_avg_kuo, mid_zeros, ppp_base11)

    p2_parts_group[12] = ppp_base12[:, :, 7:20, :]
    local_feat_avg = F.max_pool2d(p2_parts_group[12], p2_parts_group[12].size()[2:])
    local_feat_avg_kuo = local_feat_avg.repeat(1, 1, 24, 8)
    ppp_base13 = torch.where(ppp_base12 == local_feat_avg_kuo, mid_zeros, ppp_base12)

    p2_parts_group[13] = ppp_base13[:, :, 6:19, :]
    local_feat_avg = F.max_pool2d(p2_parts_group[13], p2_parts_group[13].size()[2:])
    local_feat_avg_kuo = local_feat_avg.repeat(1, 1, 24, 8)
    ppp_base14 = torch.where(ppp_base13 == local_feat_avg_kuo, mid_zeros, ppp_base13)

    p2_parts_group[14] = ppp_base14[:, :, 4:17, :]
    local_feat_avg = F.max_pool2d(p2_parts_group[14], p2_parts_group[14].size()[2:])
    local_feat_avg_kuo = local_feat_avg.repeat(1, 1, 24, 8)
    ppp_base15 = torch.where(ppp_base14 == local_feat_avg_kuo, mid_zeros, ppp_base14)

    p2_parts_group[15] = ppp_base15[:, :, 7:19, :]
    local_feat_avg = F.max_pool2d(p2_parts_group[15], p2_parts_group[15].size()[2:])
    local_feat_avg_kuo = local_feat_avg.repeat(1, 1, 24, 8)
    ppp_base16 = torch.where(ppp_base15 == local_feat_avg_kuo, mid_zeros, ppp_base15)

    p2_parts_group[16] = ppp_base16[:, :, 2:14, :]
    local_feat_avg = F.max_pool2d(p2_parts_group[16], p2_parts_group[16].size()[2:])
    local_feat_avg_kuo = local_feat_avg.repeat(1, 1, 24, 8)
    ppp_base17 = torch.where(ppp_base16 == local_feat_avg_kuo, mid_zeros, ppp_base16)

    p2_parts_group[17] = ppp_base17[:, :, 0:11, :]
    local_feat_avg = F.max_pool2d(p2_parts_group[17], p2_parts_group[17].size()[2:])
    local_feat_avg_kuo = local_feat_avg.repeat(1, 1, 24, 8)
    ppp_base18 = torch.where(ppp_base17 == local_feat_avg_kuo, mid_zeros, ppp_base17)

    p2_parts_group[18] = ppp_base18[:, :, 9:20, :]
    local_feat_avg = F.max_pool2d(p2_parts_group[18], p2_parts_group[18].size()[2:])
    local_feat_avg_kuo = local_feat_avg.repeat(1, 1, 24, 8)
    ppp_base19 = torch.where(ppp_base18 == local_feat_avg_kuo, mid_zeros, ppp_base18)

    p2_parts_group[19] = ppp_base19[:, :, 8:18, :]
    local_feat_avg = F.max_pool2d(p2_parts_group[19], p2_parts_group[19].size()[2:])
    local_feat_avg_kuo = local_feat_avg.repeat(1, 1, 24, 8)
    ppp_base20 = torch.where(ppp_base19 == local_feat_avg_kuo, mid_zeros, ppp_base19)

    p2_parts_group[20] = ppp_base20[:, :, 4:14, :]
    local_feat_avg = F.max_pool2d(p2_parts_group[20], p2_parts_group[20].size()[2:])
    local_feat_avg_kuo = local_feat_avg.repeat(1, 1, 24, 8)
    ppp_base21 = torch.where(ppp_base20 == local_feat_avg_kuo, mid_zeros, ppp_base20)

    p2_parts_group[21] = ppp_base21[:, :, 5:15, :]
    local_feat_avg = F.max_pool2d(p2_parts_group[21], p2_parts_group[21].size()[2:])
    local_feat_avg_kuo = local_feat_avg.repeat(1, 1, 24, 8)
    ppp_base22 = torch.where(ppp_base21 == local_feat_avg_kuo, mid_zeros, ppp_base21)

    p2_parts_group[22] = ppp_base22[:, :, 5:14, :]
    local_feat_avg = F.max_pool2d(p2_parts_group[22], p2_parts_group[22].size()[2:])
    local_feat_avg_kuo = local_feat_avg.repeat(1, 1, 24, 8)
    ppp_base23 = torch.where(ppp_base22 == local_feat_avg_kuo, mid_zeros, ppp_base22)

    p2_parts_group[23] = ppp_base23[:, :, 15:24, :]

    for i in range(0, 24):
        local_feat_avg = F.max_pool2d(p2_parts_group[i], p2_parts_group[i].size()[2:])
        p2_all_avg = p2_all_avg + local_feat_avg


    #h1 = p2_all_avg
    p2_all_avg = p2_all_avg.view(p2_all_avg.size(0), -1)

    '''
    p2_all_avg = p2_all_avg.view(p2_all_avg.size(0), -1)

    p2_all_avg_all = torch.sign(p2_all_avg) * torch.sqrt(torch.abs(p2_all_avg) + 1e-10)

    ##

    p2_all_avg_all = self.dense_p2_all(p2_all_avg_all)

    p2_all_score_v_2 = self.classifer_p2_all(p2_all_avg_all)
    p2_all_score_v_2 = F.log_softmax(p2_all_score_v_2, 1)
    '''


    ###
    '''
    #
    h_1 = h1.view(h1.size(0), 2048, 192)
    h1_min_2 = h_1.min(2)[0].unsqueeze(dim=2).repeat(1, 1, 192)
    h1_inter_2 = h_1 - h1_min_2
    h1_max_2 = h1_inter_2.max(2)[0].unsqueeze(dim=2).repeat(1, 1, 192) + 1e-10
    h1_inter_2 = h1_inter_2 / h1_max_2

    h_yu1_2 = h1_inter_2.max(2)[0]
    h_oo = h_yu1_2 * 0.7
    h_mm = h_yu1_2 * 0.3
    h_yu1_2 = h_yu1_2 * 0.5
    h_yu1_2 = h_yu1_2.unsqueeze(dim=2)
    h_yu1_2 = h_yu1_2.repeat(1, 1, 192) - 1e-10

    h_erase1_2 = torch.where(h1_inter_2 > h_yu1_2, mid_ones_1, mid_zeros_1)

    num1h_2 = torch.sum(h_erase1_2, 2)
    # print (num1)
    # num1 = 1 / num1

    # num1 = num1.unsqueeze(dim=1).unsqueeze(dim=1)

    h1_1_m1_2 = torch.mul(h_1, h_erase1_2)
    h1_1_m1_2 = h1_1_m1_2.sum(dim=2)
    # y1_1_m1 = torch.mul(y1_1_m1, num1)
    h1_1_m1_2 = h1_1_m1_2 / num1h_2
    #
    
    #

    h_mm = h_mm.unsqueeze(dim=2)
    h_mm = h_mm.repeat(1, 1, 192) - 1e-10

    h_mm = torch.where(h1_inter_2 > h_mm, mid_ones_1, mid_zeros_1)

    mm2 = torch.sum(h_mm, 2)
    # print (num1)
    # num1 = 1 / num1

    # num1 = num1.unsqueeze(dim=1).unsqueeze(dim=1)

    h1_mm = torch.mul(h_1, h_mm)
    h1_mm = h1_mm.sum(dim=2)
    # y1_1_m1 = torch.mul(y1_1_m1, num1)
    h1_mm = h1_mm / mm2
    #




    h1_max_2 = F.max_pool2d(h1, h1.size()[2:])
    h1_1_2 = F.avg_pool2d(h1, h1.size()[2:])
    h1_1_2 = h1_1_2 + h1_max_2
    h1_1_2 = h1_1_2.view(h1_1_2.size(0), -1)
    h1_1_2 = h1_1_2 + h1_1_m1_2 + h1_mm
    '''
    h1_1_2 = torch.sign(p2_all_avg) * torch.sqrt(torch.abs(p2_all_avg) + 1e-10)
    h1_1_2 = self.dense_h1_2(h1_1_2)
    # y1_1 = out

    # y1_2 = F.avg_pool2d(x1_2, x1_2.size()[2:])
    # y1_2 = y1_2.view(y1_2.size(0), -1)

    # y1_3 = F.avg_pool2d(x1_3, x1_3.size()[2:])
    # y1_3 = y1_3.view(y1_3.size(0), -1)

    h1_x1_score_v_2 = self.classiferh1_2(h1_1_2)
    h1_x1_score_v_2 = F.log_softmax(h1_x1_score_v_2, 1)





    y1_all = torch.cat((y1_1, y1_2, y1_1_2, y1_2_2, h1_1_2),1)
    #y1_all = torch.cat((y1_1, y1_2, y1_1_2), 1)
    #y1_all = y1_2_2

    #y1_x1_score_b = self.classifer3(y1_all)
    #y1_x1_score_b = F.log_softmax(y1_x1_score_b, 1)


    return y1_all, y1_1,  y1_x1_score_v, y1_2, y2_x1_score_v,  y1_1_2, y1_x1_score_v_2, y1_2_2, y2_x1_score_v_2, h1_1_2, h1_x1_score_v_2

