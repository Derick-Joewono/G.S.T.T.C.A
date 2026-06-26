"""
Flow : dual input siamese features : 
1. resnet + bpm = deep feature + boundary = fitur semantik dan fitur tepi obtained 
2. coarse predict = tebak awal -> simpen -> pake ntr
3. graph encoder = fitur t1 dan t2 ditumpuk ->run cfgcnhead(transformer dan gcn) = representasi grafik global dibelah jadi token 1(t1) dan token 2(t2)
4. graph decoder = terima fitur resnet+bpm -> kirim cfgcn head as decoder(decoder ini terima output proses 3 jg)
5. krm = dapet fine mask, decoder, coarse mask -> jika da conflict ->detailing spesifict location -> koreksi
6. fitur T1 dan T2 yg sudah mateng dikurangi t1-t2 utk tau selisih absolut dan upsample 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.optim import lr_scheduler
from torch.hub import load_state_dict_from_url
import functools
from einops import rearrange
import torchvision.models as vision_models

# Alias untuk kemudahan
BatchNorm2d = nn.BatchNorm2d
BatchNorm1d = nn.BatchNorm1d

# ==============================================================================
# 1. HELPER FUNCTIONS
# ==============================================================================

class TwoLayerConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1, bias=False),
                        nn.BatchNorm2d(in_channels),
                        nn.ReLU(),
                        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1)
                        )

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class EDGModule(nn.Module):
    def __init__(self, channel):
        super(EDGModule, self).__init__()
        self.relu = nn.ReLU(True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_3 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat4 = BasicConv2d(4 * channel, 4 * channel, 3, padding=1)

        self.conv5 = nn.Conv2d(4 * channel, 50, 1)

    def forward(self, x1, x2, x3):  # 16x16, 32x32, 64x64
        up_x1 = self.conv_upsample1(self.upsample(x1))
        conv_x2 = self.conv_1(x2)
        cat_x2 = self.conv_concat2(torch.cat((up_x1, conv_x2), 1))

        up_x2 = self.conv_upsample2(self.upsample(x2))
        conv_x3 = self.conv_2(x3)
        cat_x3 = self.conv_concat3(torch.cat((up_x2, conv_x3), 1))

        up_cat_x2 = self.conv_upsample3(self.upsample(cat_x2))
        conv_cat_x3 = self.conv_3(cat_x3)
        cat_x4 = self.conv_concat4(torch.cat((up_cat_x2, conv_cat_x3), 1))
        x = self.conv5(cat_x4)
        return x

def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1: 
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, init_gain=init_gain)
    return net


# ==============================================================================
# 2. KNOWLEDGE REVIEW MODULE (KRM)
# ==============================================================================

class PredictionHead(nn.Module):
    def __init__(self, channel):
        super(PredictionHead, self).__init__()
        self.mask_generation = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 1, kernel_size=1)
        )

    def forward(self, feature):
        mask = self.mask_generation(feature)
        return mask

class FeatureEnhancementUnit(nn.Module):
    def __init__(self, in_channel, channel):
        super(FeatureEnhancementUnit, self).__init__()
        self.feature_transition = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )
        hid_channel = max(8, channel // 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channel, hid_channel),
            nn.ReLU(inplace=True),
            nn.Linear(hid_channel, in_channel),
            nn.Sigmoid()
        )
        self.feature_context = nn.Sequential(
            nn.Conv2d(in_channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, mask=None):
        x = self.feature_transition(x)
        if mask is not None:
            x = x * mask + x
        B, C, _, _ = x.size()
        vec_y = self.avg_pool(x).view(B, C)
        channel_att = self.channel_attention(vec_y).view(B, C, 1, 1)
        feu_out = self.feature_context(x * channel_att)
        return feu_out

class KnowledgeReviewModule(nn.Module):
    def __init__(self, in_channel, channel):
        super(KnowledgeReviewModule, self).__init__()
        self.feu_1 = FeatureEnhancementUnit(in_channel, channel)
        self.feu_2 = FeatureEnhancementUnit(in_channel, channel)
        self.feu_3 = FeatureEnhancementUnit(in_channel, channel)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channel + 1, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, feature, fine_mask, coarse_mask):
        # without attention
        context_1 = self.feu_1(feature)
        # reverse attention
        reverse_att = 1 - torch.sigmoid(coarse_mask)
        context_2 = self.feu_2(feature, reverse_att)
        # uncertainty attention
        uncertainty_att = (1 - torch.sigmoid(coarse_mask)) * torch.sigmoid(fine_mask) + \
                          (1 - torch.sigmoid(fine_mask)) * torch.sigmoid(coarse_mask)
        context_3 = self.feu_3(feature, uncertainty_att)
        feature = context_1 + context_2 + context_3
        krm_out = self.fusion_conv(torch.cat([feature, fine_mask], dim=1))
        return krm_out


# ==============================================================================
# 3. CF-GCN MODULES
# ==============================================================================

class SpatialGCN(nn.Module):
    def __init__(self, plane):
        super(SpatialGCN, self).__init__()
        inter_plane = plane // 2
        self.node_k = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_v = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_v_y = nn.Conv2d(plane*2, plane, kernel_size=1)
        self.node_q = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_q_y = nn.Conv2d(plane*2, plane, kernel_size=1)

        self.conv_wg = nn.Conv1d(inter_plane, inter_plane, kernel_size=1, bias=False)
        self.bn_wg = BatchNorm1d(inter_plane)

        self.conv_wg_decode = nn.Conv1d(inter_plane*2, inter_plane*2, kernel_size=1, bias=False)
        self.bn_wg_decode = BatchNorm1d(inter_plane*2)

        self.softmax = nn.Softmax(dim=2)

        self.out = nn.Sequential(nn.Conv2d(inter_plane, plane, kernel_size=1),
                                 BatchNorm2d(plane))
        self.outdecode = nn.Sequential(nn.Conv2d(inter_plane*2, plane, kernel_size=1),
                                 BatchNorm2d(plane))
        self.xpre = nn.Sequential(nn.Conv2d(inter_plane*4, inter_plane*2, kernel_size=1),
                                 BatchNorm2d(inter_plane*2),
                                 nn.Conv2d(inter_plane*2, plane, kernel_size=1),
                                 BatchNorm2d(plane)
                                  )

    def forward(self, x, y):
        if y is None:
            node_k = self.node_k(x)
            node_v = self.node_v(x)
            node_q = self.node_q(x)
        else:
            node_k = y
            node_v = self.node_v_y(x)
            node_q = self.node_q_y(x)
        b, c, h, w = node_k.size()
        node_k = node_k.view(b, c, -1).permute(0, 2, 1)
        node_q = node_q.view(b, c, -1)
        node_v = node_v.view(b, c, -1).permute(0, 2, 1)
        AV = torch.bmm(node_q, node_v)
        AV = self.softmax(AV)
        AV = torch.bmm(node_k, AV)
        AV = AV.transpose(1, 2).contiguous()
        if y is None:
            AVW = self.conv_wg(AV)
            AVW = self.bn_wg(AVW)
        else:
            AVW = self.conv_wg_decode(AV)
            AVW = self.bn_wg_decode(AVW)
        AVW = AVW.view(b, c, h, -1)
        if y is None:
            out = F.relu_(self.out(AVW) + x)
        else:
            out = F.relu_(self.outdecode(AVW) + self.xpre(x))
        return out


class CFGCN(nn.Module):
    def __init__(self, planes, ratio=4):
        super(CFGCN, self).__init__()

        self.phi = nn.Conv2d(planes, planes // ratio * 2, kernel_size=1, bias=False)
        self.bn_phi = BatchNorm2d(planes // ratio * 2)
        self.theta = nn.Conv2d(planes, planes // ratio, kernel_size=1, bias=False)
        self.bn_theta = BatchNorm2d(planes // ratio)

        self.thetadecode = nn.Conv2d(planes//2, planes // ratio , kernel_size=1, bias=False)
        self.bn_thetadecode = BatchNorm2d(planes // ratio )

        self.conv_adj = nn.Conv1d(planes // ratio, planes // ratio, kernel_size=1, bias=False)
        self.bn_adj = BatchNorm1d(planes // ratio)

        self.conv_wg = nn.Conv1d(planes // ratio * 2, planes // ratio * 2, kernel_size=1, bias=False)
        self.bn_wg = BatchNorm1d(planes // ratio * 2)

        self.conv3 = nn.Conv2d(planes // ratio * 2, planes, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes)

        self.local = nn.Sequential(
            nn.Conv2d(planes, planes, 3, groups=planes, stride=2, padding=1, bias=False),
            BatchNorm2d(planes),
            nn.Conv2d(planes, planes, 3, groups=planes, stride=2, padding=1, bias=False),
            BatchNorm2d(planes),
            nn.Conv2d(planes, planes, 3, groups=planes, stride=2, padding=1, bias=False),
            BatchNorm2d(planes))
        self.localdecode = nn.Sequential(
            nn.Conv2d(planes//2, planes//2, 3, groups=planes//2, stride=2, padding=1, bias=False),
            BatchNorm2d(planes//2),
            nn.Conv2d(planes//2, planes//2, 3, groups=planes//2, stride=2, padding=1, bias=False),
            BatchNorm2d(planes//2),
            nn.Conv2d(planes//2, planes//2, 3, groups=planes//2, stride=2, padding=1, bias=False),
            BatchNorm2d(planes//2))
        self.gcn_local_attention = SpatialGCN(planes)
        self.gcn_local_attentiondecode = SpatialGCN(planes//2)

        self.final = nn.Sequential(nn.Conv2d(planes * 2, planes, kernel_size=1, bias=False),
                                   BatchNorm2d(planes))
        self.finaldecode = nn.Sequential(nn.Conv2d(192, planes, kernel_size=1, bias=False),
                                   BatchNorm2d(planes))

    def to_matrix(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c, -1)
        return x

    def forward(self, feat, featdecode):
        x = feat
        if featdecode is None:
            local = self.local(feat)
            local = self.gcn_local_attention(local, None)
        else:
            localtoken = self.localdecode(featdecode)
            localfeat = self.local(feat)
            local = self.gcn_local_attentiondecode(localfeat, localtoken)
        if featdecode is None:
            local = F.interpolate(local, size=x.size()[2:], mode='bilinear', align_corners=True)
            spatial_local_feat = x * local + x
        else:
            local = F.interpolate(local, size=featdecode.size()[2:], mode='bilinear', align_corners=True)
            spatial_local_feat = featdecode * local + featdecode

        x_sqz, b = x, x
        x_sqz = self.phi(x_sqz)
        x_sqz = self.bn_phi(x_sqz)
        x_sqz = self.to_matrix(x_sqz)

        if featdecode is None:
            b = self.theta(b)
            b = self.bn_theta(b)
            b = self.to_matrix(b)
        else:
            b = self.thetadecode(featdecode)
            b = self.bn_thetadecode(b)
            b = self.to_matrix(b)
            
        z_idt = torch.matmul(x_sqz, b.transpose(1, 2))

        z = z_idt.transpose(1, 2).contiguous()
        z = self.conv_adj(z)
        z = self.bn_adj(z)

        z = z.transpose(1, 2).contiguous()
        z = z_idt + z

        z = self.conv_wg(z)
        z = self.bn_wg(z)

        y = torch.matmul(z, b)
        n, _, h, w = x.size()
        y = y.view(n, -1, h, w)

        y = self.conv3(y)
        y = self.bn3(y)

        g_out = F.relu_(x + y)

        if featdecode is None:
            out = self.final(torch.cat((spatial_local_feat, g_out), 1))
        else:
            out = self.finaldecode(torch.cat((spatial_local_feat, g_out), 1))

        return out


class CFGCNHead(nn.Module):
    def __init__(self, inplanes, interplanes, num_classes):
        super(CFGCNHead, self).__init__()
        self.conva = nn.Sequential(nn.Conv2d(inplanes, interplanes, 3, padding=1, bias=False),
                                BatchNorm2d(interplanes),
                                nn.ReLU(interplanes))
        self.dualgcn = CFGCN(interplanes)
        self.convb = nn.Sequential(nn.Conv2d(interplanes, interplanes//2, 3, padding=1, bias=False),
                                BatchNorm2d(interplanes//2),
                                nn.ReLU(interplanes//2))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(inplanes + interplanes, interplanes, kernel_size=3, padding=1, dilation=1, bias=False),
            BatchNorm2d(interplanes),
            nn.ReLU(interplanes),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x, y):
        if y is None:
            output = self.conva(x)
            output = self.dualgcn(output, None)
        else:
            output = self.conva(x)
            output = self.dualgcn(output, y)
        output = self.convb(output)
        return output


# ==============================================================================
# 4. BACKBONE & MAIN ARCHITECTURE
# ==============================================================================

class CFGCN_ResNet(torch.nn.Module):
    def __init__(self, input_nc, output_nc,
                 resnet_stages_num=5, backbone='resnet50', 
                 output_sigmoid=False, if_upsample_2x=True, edgechannel=4):
        
        super(CFGCN_ResNet, self).__init__()
        expand = 1

        if backbone == 'resnet18':
            self.resnet = vision_models.resnet18(pretrained=True,
                                          replace_stride_with_dilation=[False, True, True])
        elif backbone == 'resnet34':
            self.resnet = vision_models.resnet34(pretrained=True,
                                          replace_stride_with_dilation=[False, True, True])
        elif backbone == 'resnet50':
            self.resnet = vision_models.resnet50(pretrained=False, 
                                          replace_stride_with_dilation=[False, True, True])
            expand = 4
        else:
            raise NotImplementedError

        # MODIFIKASI: UBAH INPUT CHANNEL MENJADI 10
        self.resnet.conv1 = nn.Conv2d(
            in_channels=input_nc,  
            out_channels=64,       
            kernel_size=7, 
            stride=2, 
            padding=3, 
            bias=False
        )
        
        nn.init.kaiming_normal_(self.resnet.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2)
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.downsamplex2 = nn.Conv2d(256, 256, kernel_size=2, stride=2)
        self.downsamplex2_1 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=2, stride=2),
                                            nn.BatchNorm2d(32),
                                            nn.ReLU())
        self.classifier = TwoLayerConv2d(in_channels=64, out_channels=output_nc)

        self.resnet_stages_num = resnet_stages_num
        self.if_upsample_2x = if_upsample_2x
        if self.resnet_stages_num == 5:
            layers = 512 * expand
        elif self.resnet_stages_num == 4:
            layers = 256 * expand
        elif self.resnet_stages_num == 3:
            layers = 128 * expand
        else:
            raise NotImplementedError
        self.conv_pred = nn.Conv2d(layers+50, 128, kernel_size=3, padding=1)

        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

        channel = 32
        self.rfb2_1 = BasicConv2d(64, channel, 1)
        self.rfb3_1 = BasicConv2d(256, channel, 1)
        self.rfb4_1 = BasicConv2d(512, channel, 1)
        self.edge = EDGModule(channel)

        self.pre = BasicConv2d(1024, 256, 1)

    def forward(self, x1, x2):
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x1)
        x = torch.abs(x1 - x2)
        if not self.if_upsample_2x:
            x = self.upsamplex2(x)
        x = self.upsamplex4(x)
        x = self.classifier(x)

        if self.output_sigmoid:
            x = self.sigmoid(x)
        return x

    def forward_single(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)

        x1_rfb = self.rfb2_1(x)
        x = self.resnet.maxpool(x)
        x_4 = self.resnet.layer1(x)  

        x2_rfb = self.rfb3_1(x_4)
        x_8 = self.resnet.layer2(x_4)  

        x3_rfb = self.rfb4_1(x_8)

        if self.resnet_stages_num > 3:
            x_8 = self.resnet.layer3(x_8)  
        if self.resnet_stages_num == 5:
            x_8 = self.resnet.layer4(x_8)  
        elif self.resnet_stages_num > 5:
            raise NotImplementedError

        edge_feat = self.edge(x3_rfb, x2_rfb, x1_rfb)
        alledges = F.interpolate(edge_feat, size=(32, 32), mode='bilinear', align_corners=True)

        x_8 = self.pre(x_8)
        return x_8, alledges, x3_rfb, x2_rfb, x1_rfb


class BASE_GCN(CFGCN_ResNet):
    def __init__(self, input_nc, output_nc,
                 resnet_stages_num=4,
                 if_upsample_2x=True,
                 pool_size=2,
                 backbone='resnet50'):
        super(BASE_GCN, self).__init__(input_nc, output_nc, backbone=backbone,
                                               resnet_stages_num=resnet_stages_num,
                                               if_upsample_2x=if_upsample_2x,
                                               )
        self.pooling_size = pool_size
        self.cfgcn = CFGCNHead(562, 256, num_classes=64)
        self.cfgcndecode = CFGCNHead(306, 128, num_classes=32)
        self.conv_c = nn.Conv2d(256, 32, kernel_size=3, padding=1)
        self.krm = KnowledgeReviewModule(320, 64)
        self.krmtoken = KnowledgeReviewModule(128, 64)
        self.mask_generation = PredictionHead(64)
        self.coarse_mask_generation = PredictionHead(256)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1, alledges1, A3, A2, A1 = self.forward_single(x1)
        x2, alledges2, B3, B2, B1 = self.forward_single(x2)

        x1_coarse = x1
        x2_coarse = x2
        x1_coarse_mask = self.coarse_mask_generation(x1)
        x2_coarse_mask = self.coarse_mask_generation(x2)

        edge_abs = self.edge(A3-B3, A2-B2, A1-B1)
        alledge_abs = F.interpolate(edge_abs, size=(32, 32), mode='bilinear', align_corners=True)

        self.tokens_ = torch.cat([x1, x2, alledge_abs], dim=1)
        self.tokens = self.cfgcn(self.tokens_, None)

        token1, token2 = self.tokens.chunk(2, dim=1)
        x1 = torch.cat([x1, alledges1], dim=1)
        x2 = torch.cat([x2, alledges2], dim=1)

        x1 = self.cfgcndecode(x1, token1)
        x2 = self.cfgcndecode(x2, token2)

        x1_mask = self.mask_generation(x1)
        x2_mask = self.mask_generation(x2)

        token1_mask = self.mask_generation(token1)
        token2_mask = self.mask_generation(token2)

        x1_krm_corase = self.krm(torch.cat((x1, x1_coarse), dim=1), x1_mask, x1_coarse_mask)
        x1_krm_token = self.krmtoken(torch.cat((x1, token1), dim=1), x1_mask, token1_mask)
        x1_krm = self.fusion_conv(torch.cat((x1_krm_corase, x1_krm_token), dim=1))

        x2_krm_corase = self.krm(torch.cat((x2, x2_coarse), dim=1), x2_mask, x2_coarse_mask)
        x2_krm_token = self.krmtoken(torch.cat((x2, token2), dim=1), x2_mask, token2_mask)
        x2_krm = self.fusion_conv(torch.cat((x2_krm_corase, x2_krm_token), dim=1))

        x = torch.abs(x1_krm - x2_krm)
        x = self.upsamplex2(x)
        x = self.upsamplex4(x)
        x = self.classifier(x)
        if self.output_sigmoid:
            x = self.sigmoid(x)
        return x

# ==============================================================================
# 5. LOSS FUNCTION ADAPTER
# ==============================================================================

class CFGCN_OriginalCELoss(torch.nn.Module):
    def __init__(self, raw_weights=None):
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss(weight=raw_weights, ignore_index=-1)

    def forward(self, pred, target):
        loss = self.ce(pred, target)
        return loss, {
            "ce": loss.item(), 
            "focal": 0.0, 
            "dice": 0.0,
            "boundary": 0.0
        }