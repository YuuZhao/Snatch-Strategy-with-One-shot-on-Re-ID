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
import  codecs
from  common_tool import *

def main(args):
    # 声明动态绘图器
    gd = gif_drawer()


    cudnn.benchmark = True
    cudnn.enabled = True

    # get all the labeled and unlabeled data for training
    dataset_all = datasets.create(args.dataset, osp.join(args.data_dir, args.dataset))
    l_data, u_data = get_one_shot_in_cam1(dataset_all, load_path="./examples/oneshot_{}_used_in_paper.pickle".format(
        dataset_all.name))


    # 输出该轮训练关键的提示信息
    print("{} training begin with dataset:{},batch_size:{},epoch:{},step_size:{},max_frames:{},percent:{}".format(args.exp_name,args.dataset,args.batch_size,args.epoch,args.step_size,args.max_frames,args.percent))

    # 指定输出文件
    # 第三部分要说明关键参数的设定
    sys.stdout = Logger(osp.join(args.logs_dir, args.dataset, args.exp_name, args.exp_order,
                                 'log_mSA' + time.strftime(".%m_%d_%H-%M-%S") + '.txt'))
    data_file = codecs.open(osp.join(args.logs_dir, args.dataset, args.exp_name, args.exp_order, 'data.txt'), mode='a')

    time_file = codecs.open(osp.join(args.logs_dir, args.dataset, args.exp_name, args.exp_order, 'time.txt'),
                                mode='a')
    save_path = osp.join(args.logs_dir, args.dataset, args.exp_name, args.exp_order)

    # initial the EUG algorithm
    eug = EUG(model_name=args.arch, batch_size=args.batch_size, mode=args.mode, num_classes=dataset_all.num_train_ids,
              data_dir=dataset_all.images_dir, l_data=l_data, u_data=u_data, save_path=save_path,
              max_frames=args.max_frames)
    # 开始的时间记录
    exp_start = time.time()
    eug.resume(osp.join(save_path,'Dissimilarity_step_0.ckpt'),0)
    pred_y, pred_score, label_pre, id_num = eug.estimate_label()
    select_num = math.ceil(len(u_data) * args.percent) # 用于训练tagper的数量
    selected_idx = eug.select_top_data(pred_score,select_num)
    train_data, select_pre = eug.generate_new_train_data(selected_idx,pred_y)
    eug.train(train_data,1,epochs=args.epoch, step_size=args.step_size, init_lr=0.1)
    pred_y, pred_score, label_pre, id_num = eug.estimate_label()   #tagper的数据
    data_file.write('percent:{} label_pre:{}'.format(args.percent,label_pre))
    select_pre =[5,10,20,30,40]
    for sp in select_pre: #对采样比例做便利\
        select_num =math.ceil(len(u_data) * sp /100)
        selected_idx = eug.select_top_data(pred_score,select_num)
        _, select_pre = eug.generate_new_train_data(selected_idx, pred_y)
        data_file.write(' sp{}:{}'.format(sp,select_pre))
    data_file.write('\n')

    data_file.close()
    if (args.clock):
        exp_end = time.time()
        exp_time = exp_end - exp_start
        h, m, s = changetoHSM(exp_time)
        print("experiment is over, cost %02d:%02d:%02.6f" % ( h, m, s))
        time_file.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Snatch Strategy')
    parser.add_argument('-d', '--dataset', type=str, default='DukeMTMC-VideoReID',choices=datasets.names())  #DukeMTMC-VideoReID
    parser.add_argument('-b', '--batch-size', type=int, default=16)
    parser.add_argument('--epoch',type=int,default=70)
    parser.add_argument('--step_size',type=int,default=55)
    parser.add_argument('--percent', type=float, default=0)  # 第二次加进去的量
    working_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--data_dir', type=str, metavar='PATH',default=os.path.join(working_dir, 'data'))  # 加载数据集的根目录
    parser.add_argument('--logs_dir', type=str, metavar='PATH',default=os.path.join(working_dir, 'logs'))  # 保持日志根目录
    parser.add_argument('--exp_name',type=str,default="ero")
    parser.add_argument('--exp_order',type=str,default="1")
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--mode', type=str, choices=["Classification", "Dissimilarity"], default="Dissimilarity")   #这个考虑要不要取消掉
    parser.add_argument('--max_frames', type=int, default=100)
    parser.add_argument('--clock',type=bool, default=True)  #是否记时
    parser.add_argument('--gdraw',type=bool, default=False)  #是否实时绘图

    #下面是暂时不知道用来做什么的参数
    parser.add_argument('-a', '--arch', type=str, default='avg_pool',choices=models.names())  #eug model_name
    parser.add_argument('-i', '--iter-step', type=int, default=5)
    parser.add_argument('-g', '--gamma', type=float, default=0.3)
    parser.add_argument('-l', '--l', type=float)
    parser.add_argument('--continuous', action="store_true")
    main(parser.parse_args())
