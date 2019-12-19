from __future__ import absolute_import

from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn import init
import torch
import torchvision
import math

from .resnet import *


__all__ = ["End2End_AvgPooling"]


class APM(nn.Module):
    def __init__(self, feat_size, n_fc=224):
        super(self.__class__, self).__init__()
        self.fc = nn.Linear(feat_size, n_fc)
        init.normal(self.fc.weight, std=0.001)
        init.constant(self.fc.bias, 0)

    def forward(self, x):
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class AvgPooling(nn.Module):
    def __init__(self, input_feature_size, num_classes, mode, embeding_fea_size=1024, dropout=0.5):
        super(self.__class__, self).__init__()

        is_output_feature = {"Dissimilarity": True, "Classification": False}
        self.is_output_feature = is_output_feature[mode]

        # embeding
        self.embeding = nn.Linear(input_feature_size, embeding_fea_size)
        self.embeding_bn = nn.BatchNorm1d(embeding_fea_size)
        init.kaiming_normal(self.embeding.weight, mode='fan_out')
        init.constant(self.embeding.bias, 0)
        init.constant(self.embeding_bn.weight, 1)
        init.constant(self.embeding_bn.bias, 0)
        self.drop = nn.Dropout(dropout)

        # classifier
        self.classify_fc = nn.Linear(embeding_fea_size, num_classes)
        init.normal(self.classify_fc.weight, std=0.001)
        init.constant(self.classify_fc.bias, 0)

    def forward(self, inputs):
        # avg_pool_feat_g = inputs[0].mean(dim=1)
        # avg_pool_feat_l1 = inputs[1].mean(dim=1)
        # avg_pool_feat_l2 = inputs[2].mean(dim=1)
        # avg_pool_feat_l3 = inputs[3].mean(dim=1)
        # avg_pool_feat_l4 = inputs[4].mean(dim=1)

        # avg_pool_feat = torch.cat([avg_pool_feat_g, avg_pool_feat_l1,
        #                            avg_pool_feat_l2, avg_pool_feat_l3, avg_pool_feat_l4], dim=1)

        avg_pool_feat = inputs.mean(dim=1)
        # avg_pool_feat = torch.median(inputs,dim=1)[0]
        if (not self.training) and self.is_output_feature:
            return F.normalize(avg_pool_feat, p=2, dim=1)

        # embeding
        net = self.drop(avg_pool_feat)
        net = self.embeding(net)
        net = self.embeding_bn(net)
        net = F.relu(net)

        net = self.drop(net)

        # classifier
        predict = self.classify_fc(net)
        return predict


class End2End_AvgPooling(nn.Module):  # 训练的基本模型在这儿

    def __init__(self, pretrained=True, dropout=0, num_classes=0, mode="retrieval"):
        super(self.__class__, self).__init__()  # 不明白这个是定义来干什么的,init里面为空

        self.CNN = resnet50(dropout=dropout, cut_at_pooling=True)

        self.apm_g = APM(2048, 224)
        self.apm_l1 = APM(341, 200)
        self.apm_l2 = APM(341*2, 200)
        self.apm_l3 = APM(341*2, 200)
        self.apm_l4 = APM(343, 200)

        # self.avg_pooling = AvgPooling(input_feature_size=2048, num_classes=num_classes, dropout=dropout, mode=mode)
        self.avg_pooling_g = AvgPooling(
            input_feature_size=224, num_classes=num_classes, dropout=dropout, mode=mode)
        self.avg_pooling_l1 = AvgPooling(
            input_feature_size=200, num_classes=num_classes, dropout=dropout, mode=mode)
        self.avg_pooling_l2 = AvgPooling(
            input_feature_size=200, num_classes=num_classes, dropout=dropout, mode=mode)
        self.avg_pooling_l3 = AvgPooling(
            input_feature_size=200, num_classes=num_classes, dropout=dropout, mode=mode)
        self.avg_pooling_l4 = AvgPooling(
            input_feature_size=200, num_classes=num_classes, dropout=dropout, mode=mode)

    def forward(self, x):
        assert len(x.data.shape) == 5
        # reshape (batch, samples, ...) ==> (batch * samples, ...)
        oriShape = x.data.shape
        x = x.view(-1, oriShape[2], oriShape[3], oriShape[4])

        # resnet encoding
        resnet_feature = self.CNN(x)

        feat_g = self.apm_g(resnet_feature)
        feat_l1 = self.apm_l1(resnet_feature[:, :341])
        feat_l2 = self.apm_l2(resnet_feature[:, 341:341*3])
        feat_l3 = self.apm_l3(resnet_feature[:, 341*3:341*5])
        feat_l4 = self.apm_l4(resnet_feature[:, 341*5:])

        feat_g = feat_g.view(oriShape[0], oriShape[1], -1)
        feat_l1 = feat_l1.view(oriShape[0], oriShape[1], -1)
        feat_l2 = feat_l2.view(oriShape[0], oriShape[1], -1)
        feat_l3 = feat_l3.view(oriShape[0], oriShape[1], -1)
        feat_l4 = feat_l4.view(oriShape[0], oriShape[1], -1)

        if (not self.avg_pooling_g.training) and self.avg_pooling_g.is_output_feature:
            avg_pool_feat = torch.cat([
                feat_g.mean(dim=1),
                feat_l1.mean(dim=1),
                feat_l2.mean(dim=1),
                feat_l3.mean(dim=1),
                feat_l4.mean(dim=1)], dim=1)
            return F.normalize(avg_pool_feat, p=2, dim=1)

        # # reshape back into (batch, samples, ...)
        # resnet_feature = resnet_feature.view(oriShape[0], oriShape[1], -1)
        #
        # # avg pooling
        # # if eval and cut_off_before_logits, return predict;  else return avg pooling feature
        # predict = self.avg_pooling(
        #     [feat_g, feat_l1, feat_l2, feat_l3, feat_l4])
        pred_g = self.avg_pooling_g(feat_g)
        pred_l1 = self.avg_pooling_l1(feat_l1)
        pred_l2 = self.avg_pooling_l2(feat_l2)
        pred_l3 = self.avg_pooling_l3(feat_l3)
        pred_l4 = self.avg_pooling_l4(feat_l4)

        return [pred_g, pred_l1, pred_l2, pred_l3, pred_l4]
