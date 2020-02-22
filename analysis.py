from __future__ import print_function, absolute_import
from reid.snatch import *
from reid import datasets
from reid import models
import numpy as np
import torch
import argparse
import os

from reid.utils.logging import Logger
import os.path as osp
import sys
from torch.backends import cudnn
from reid.utils.serialization import load_checkpoint
from torch import nn
import time
import math
import pickle
import time

import matplotlib.pyplot as plt

import os
import codecs

from common_tool import *


def main(args):
    cudnn.benchmark = True
    cudnn.enabled = True

    # get all the labeled and unlabeled data for training
    dataset_all = datasets.create(args.dataset, osp.join(args.data_dir, args.dataset))
    l_data, u_data = get_one_shot_in_cam1(dataset_all, load_path="./examples/oneshot_{}_used_in_paper.pickle".format(
        dataset_all.name))  # 分别返回 带标注数据 和 未标注数据
    # 指定输出文件
    # 第三部分要说明关键参数的设定
    sys.stdout = Logger(osp.join(args.logs_dir, args.dataset, args.exp_name, args.exp_order,
                                 'log' + time.strftime(".%m_%d_%H-%M-%S") + '.txt'))
    input_list = osp.join(args.logs_dir, args.dataset, args.exp_name, args.exp_order)

    output_list = osp.join(args.logs_dir, args.dataset, args.exp_name, args.exp_order, 'analysis')

    # initial the EUG algorithm
    eug = EUG(model_name=args.arch, batch_size=args.batch_size, mode=args.mode, num_classes=dataset_all.num_train_ids,
              data_dir=dataset_all.images_dir, l_data=l_data, u_data=u_data, save_path=output_list,
              max_frames=args.max_frames)

    files = os.listdir(input_list)
    files.sort()
    for filename in files:
        if not os.path.isfile(osp.join(input_list, filename)):   #不是文件
            continue
        type = filename.split('.')[1]
        if type == 'ckpt': #是模型文件
            step = filename.split('.')[0].split('_step_')[1]
            eug.resume(osp.join(input_list,'Dissimilarity_step_{}.ckpt'.format(step)), step)
            pred_labels, true_labels, dists, acc_list, vari = eug.estimate_label_for_analysis()  #这里返回的分数是负数
            np.save(osp.join(output_list,'pred_labels','pred_labels{}.npy'.format(step)),pred_labels)
            np.save(osp.join(output_list,'true_labels','true_labels{}.npy'.format(step)),true_labels)
            np.save(osp.join(output_list,'dists','dists{}.npy'.format(step)),dists)
            np.save(osp.join(output_list,'acc_list','acc_list{}.npy'.format(step)),acc_list)
            np.save(osp.join(output_list,'vari','vari{}.npy'.format(step)),vari)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Snatch Strategy')
    parser.add_argument('-d', '--dataset', type=str, default='DukeMTMC-VideoReID',
                        choices=datasets.names())  # DukeMTMC-VideoReID \mars
    parser.add_argument('-b', '--batch-size', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=40)
    parser.add_argument('--step_size', type=int, default=30)
    parser.add_argument('--EF', type=float, default=10)  # 渐进采样系数
    parser.add_argument('--q', type=float, default=1)  # 渐进采样指数
    parser.add_argument('--amp', type=float, default=0.8)  # 方差的筛选范围.
    working_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--data_dir', type=str, metavar='PATH', default=os.path.join(working_dir, 'data'))  # 加载数据集的根目录
    parser.add_argument('--logs_dir', type=str, metavar='PATH', default=os.path.join(working_dir, 'logs'))  # 保持日志根目录
    parser.add_argument('--exp_name', type=str, default="epsm")
    parser.add_argument('--exp_order', type=str, default="0")
    parser.add_argument('--resume', type=bool, default=True)
    parser.add_argument('--mode', type=str, choices=["Classification", "Dissimilarity"],
                        default="Dissimilarity")  # 这个考虑要不要取消掉
    parser.add_argument('--max_frames', type=int, default=400)
    parser.add_argument('--clock', type=bool, default=True)  # 是否记时
    parser.add_argument('--gdraw', type=bool, default=False)  # 是否实时绘图
    parser.add_argument('--sp', type=int, default=4)
    parser.add_argument('--ts', type=int, default=11)

    # 下面是暂时不知道用来做什么的参数
    parser.add_argument('-a', '--arch', type=str, default='avg_pool', choices=models.names())  # eug model_name
    parser.add_argument('-i', '--iter-step', type=int, default=5)
    parser.add_argument('-g', '--gamma', type=float, default=0.3)
    parser.add_argument('-l', '--l', type=float)
    parser.add_argument('--continuous', action="store_true")
    main(parser.parse_args())

    '''
    文件描述 :
    用于生成制定实验数据集,生成的dist,acc_list,vari list.其中acc_list 标签估计正误文件(bool列表,.npy文件)
    运行命令如:
    python3.6  analysis.py --exp_name gradually_11step --exp_order 0
    python3.6  analysis.py --exp_name gradually_11step --exp_order 0  --dataset mars  --max_frames 100
    '''
