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

from common_tool import *

import os
import  codecs
def main(args):
    cudnn.benchmark = True
    cudnn.enabled = True

    # get all the labeled and unlabeled data for training
    dataset_all = datasets.create(args.dataset, osp.join(args.data_dir, args.dataset))
    l_data, u_data = get_one_shot_in_cam2(dataset_all, load_path="./examples/oneshot_{}_used_in_paper.pickle".format(
        dataset_all.name))
    NN = len(l_data) + len(u_data)

    # 总的训练step数的计算
    total_step = math.ceil(math.pow((100 / args.EF), (1 / args.q)))  # 这里应该取上限或者 +2  多一轮进行one-shot训练的  # EUG base 采样策略
    # total_step = math.ceil((2 * NN * args.step_s + args.yita + len(u_data)) / (args.yita + NN + len(l_data))) + 2 # big start 策略

    # 输出该轮训练关键的提示信息
    print(
        "{} training begin with dataset:{},batch_size:{},epoch:{},step_size:{},max_frames:{},total_step:{},EF:{},q:{},yita:{},step_s:{}".format(
            args.exp_name, args.dataset, args.batch_size, args.epoch, args.step_size, args.max_frames, total_step + 1,
            args.EF, args.q, args.yita, args.step_s))

    # 指定输出文件
    # 第三部分要说明关键参数的设定
    sys.stdout = Logger(osp.join(args.logs_dir, args.dataset, args.exp_name, args.exp_order,
                                 'log' + time.strftime(".%m_%d_%H-%M-%S") + '.txt'))
    data_file = codecs.open(osp.join(args.logs_dir, args.dataset, args.exp_name, args.exp_order, 'data.txt'), mode='a')
    if args.clock:
        time_file = codecs.open(osp.join(args.logs_dir, args.dataset, args.exp_name, args.exp_order, 'time.txt'),
                                mode='a')
    save_path = osp.join(args.logs_dir, args.dataset, args.exp_name, args.exp_order)

    resume_step, ckpt_file = -1, ''
    if args.resume:  # 重新训练的时候用
        resume_step, ckpt_file = resume(args)

        # initial the EUG algorithm
    eug = EUG(model_name=args.arch, batch_size=1024, mode=args.mode, num_classes=dataset_all.num_train_ids,
              data_dir=dataset_all.images_dir, l_data=l_data, u_data=u_data, save_path=args.logs_dir,
              max_frames=args.max_frames)

    data_set = 'mars'
    group_name = '03'
    middle = math.ceil(len(l_data) / 2)
    l_data_1 = l_data[:middle]
    l_data_2 = l_data[middle:]
    eug.resume('tsne/Dissimilarity_step_1.ckpt', 1)
    print('Extracting features...')
    fts, lbs, cams = eug.get_feature_with_labels_cams(l_data_1)
    print('Saving fts1...')
    np.save('tsne/{}/{}/fts_1.npy'.format(data_set,group_name), fts)
    print('Saving lbs1...')
    np.save('tsne/{}/{}/lbs_1.npy'.format(data_set,group_name), lbs)
    print('Saving cams1...')
    np.save('tsne/{}/{}/cams_1.npy'.format(data_set,group_name), cams)
    fts, lbs, cams = eug.get_feature_with_labels_cams(l_data_2)
    print('Saving fts1...')
    np.save('tsne/{}/{}/fts_2.npy'.format(data_set,group_name), fts)
    print('Saving lbs1...')
    np.save('tsne/{}/{}/lbs_2.npy'.format(data_set,group_name), lbs)
    print('Saving cams1...')
    np.save('tsne/{}/{}/cams_2.npy'.format(data_set,group_name), cams)
    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Snatch Strategy')
    parser.add_argument('-d', '--dataset', type=str, default='mars', choices=datasets.names())  # s
    parser.add_argument('-b', '--batch-size', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--step_size', type=int, default=25)
    parser.add_argument('--EF', type=float, default=5)  # 渐进采样系数
    parser.add_argument('--q', type=float, default=1)  # 渐进采样指数
    parser.add_argument('--yita', type=int, default=100)  # big start based number
    parser.add_argument('--step_s', type=int, default=10)  # big start 饱和度控制
    working_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--data_dir', type=str, metavar='PATH', default=os.path.join(working_dir, 'data'))  # 加载数据集的根目录
    parser.add_argument('--logs_dir', type=str, metavar='PATH', default=os.path.join(working_dir, 'logs'))  # 保持日志根目录
    parser.add_argument('--exp_name', type=str, default="test")
    parser.add_argument('--exp_order', type=str, default="1")
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--mode', type=str, choices=["Classification", "Dissimilarity"],
                        default="Dissimilarity")  # 这个考虑要不要取消掉
    parser.add_argument('--max_frames', type=int, default=100)
    parser.add_argument('--clock', type=bool, default=True)  # 是否记时
    parser.add_argument('--gdraw', type=bool, default=False)  # 是否实时绘图

    # 下面是暂时不知道用来做什么的参数
    parser.add_argument('-a', '--arch', type=str, default='avg_pool', choices=models.names())  # eug model_name
    parser.add_argument('-i', '--iter-step', type=int, default=5)
    parser.add_argument('-g', '--gamma', type=float, default=0.3)
    parser.add_argument('-l', '--l', type=float)
    parser.add_argument('--continuous', action="store_true")
    main(parser.parse_args())
