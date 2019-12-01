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
from common_tool import *

def main(args):

    cudnn.benchmark = True
    cudnn.enabled = True

    # get all the labeled and unlabeled data for training
    dataset_all = datasets.create(args.dataset, osp.join(args.data_dir, args.dataset))
    l_data, u_data = get_one_shot_in_cam1(dataset_all, load_path="./examples/oneshot_{}_used_in_paper.pickle".format(
        dataset_all.name))
    NN = len(l_data) + len(u_data)
    one_shot = l_data  # 初始化带标注的样本为oneshot  全过程中 l_data不做修改

    total_step = args.total_step
    add_num = math.ceil(len(u_data)/total_step)   # 最后一轮必定不足add_num的数量


    # 输出该轮训练关键的提示信息
    print("{} training begin with dataset:{},batch_size:{},epoch:{},step_size:{},max_frames:{},total_step:{},add {} sample each step.".format(args.exp_name,args.dataset,args.batch_size,args.epoch,args.step_size,args.max_frames,total_step,add_num))

    # 指定输出文件
    # 第三部分要说明关键参数的设定
    reid_path = osp.join(args.logs_dir, args.dataset,args.exp_name,args.exp_order)
    sys.stdout = Logger(osp.join(reid_path,'log'+time.strftime(".%m_%d_%H-%M-%S")+'.txt'))
    data_file =codecs.open(osp.join(reid_path,'data.txt'),mode='a')
    time_file =codecs.open(osp.join(reid_path,'time.txt'),mode='a')
    # save_path = reid_path
    tagper_path = osp.join(reid_path,'tagper')
    tagper_file = codecs.open(osp.join(tagper_path,"tagper_data.txt"),mode='a')

    resume_step, ckpt_file = -1, ''
    if args.resume:  # 重新训练的时候用
        resume_step, ckpt_file = resume(args)

    # initial the EUG algorithm
    reid = EUG(model_name=args.arch, batch_size=args.batch_size, mode=args.mode, num_classes=dataset_all.num_train_ids,
              data_dir=dataset_all.images_dir, l_data=l_data, u_data=u_data, save_path=reid_path,
              max_frames=args.max_frames)

    # 开始的时间记录
    exp_start = time.time()
    for step in range(total_step+1):
        print("{} training begin with dataset:{},batch_size:{},epoch:{},step:{}/{} saved to {}.".format(args.exp_name,args.dataset,args.batch_size, args.epoch,step+1,total_step+1,reid_path))
        print("key parameters contain add_num:{} len(one_shot):{},len(u_data):{}".format(add_num,len(one_shot),len(u_data)))

        # 开始训练
        reid_start = time.time()
        reid.train(one_shot, step, tagper=0,epochs=args.epoch, step_size=args.step_size, init_lr=0.1) if step != resume_step else reid.resume(ckpt_file, step)
        # 开始评估
        # mAP, top1, top5, top10, top20 = 0,0,0,0,0
        mAP,top1,top5,top10,top20 = reid.evaluate(dataset_all.query, dataset_all.gallery)
        # 标签估计
        # pred_y, pred_score, label_pre, id_num = 0,0,0,0
        pred_y, pred_score, label_pre, id_num = reid.estimate_label(u_data,l_data) #针对u_data进行标签估计
        reid_end = time.time()
        data_file.write(
            "step:{} mAP:{:.2%} top1:{:.2%} top5:{:.2%} top10:{:.2%} top20:{:.2%} len(one_shot):{} label_pre:{:.2%}\n".format(
                int(step + 1), mAP, top1, top5, top10, top20, len(one_shot), label_pre))
        print(
            "step:{} mAP:{:.2%} top1:{:.2%} top5:{:.2%} top10:{:.2%} top20:{:.2%} len(one_shot):{} label_pre:{:.2%}\n".format(
                int(step + 1), mAP, top1, top5, top10, top20, len(one_shot), label_pre))

        if len(u_data)==0:
            continue

        tagper_start = time.time()
        tapger = reid  #对模型进行拷贝
        v =np.ones(len(u_data))
        selected_idx = v.astype('bool')
        new_train_data, label_pre_befor = tapger.generate_new_train_data(selected_idx, pred_y,u_data)   #这个选择准确率应该是和前面的label_pre是一样的.
        train_data = new_train_data+one_shot
        tapger.train(train_data,step,tagper=1,epochs=args.epoch, step_size=args.step_size, init_lr=0.1)
        mAP, top1, top5, top10, top20 = tapger.evaluate(dataset_all.query, dataset_all.gallery)
        pred_y, pred_score, label_pre_after, id_num = tapger.estimate_label(u_data,l_data)

        #下面正对 reid 移动数据.
        if len(u_data) < add_num:
            add_num = len(u_data)
        selected_idx = reid.select_top_data(pred_score,add_num)
        one_shot, u_data , select_pre= reid.move_unlabel_to_label(selected_idx,pred_y,u_data,one_shot)
        tapger_end = time.time()

        tagper_file.write(
            "step:{} mAP:{:.2%} top1:{:.2%} top5:{:.2%} top10:{:.2%} top20:{:.2%} len(one_shot):{} label_pre_befor:{:.2%} label_pre_after:{:.2%} select_pre:{}\n".format(
                int(step + 1), mAP, top1, top5, top10, top20, len(one_shot), label_pre_befor,label_pre_after,select_pre))
        print(
            "step:{} mAP:{:.2%} top1:{:.2%} top5:{:.2%} top10:{:.2%} top20:{:.2%} len(one_shot):{} label_pre_befor:{:.2%} label_pre_after:{:.2%} select_pre:{}\n".format(
                int(step + 1), mAP, top1, top5, top10, top20, len(one_shot), label_pre_befor, label_pre_after,
                select_pre))


        if args.clock:
            reid_time = reid_end -reid_start
            tagper_time = tapger_end-tagper_start
            step_time = tapger_end+reid_start
            time_file.write("step:{}  reid_time:{} tagper_time:{} step_time:{}\n".format(int(step+1),reid_time,tagper_time,step_time))


    data_file.close()
    tagper_file.close()
    if (args.clock):
        exp_end = time.time()
        exp_time = exp_end - exp_start
        h, m, s = changetoHSM(exp_time)
        print("experiment is over, cost %02d:%02d:%02.6f" % ( h, m, s))
        time_file.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Snatch Strategy')
    parser.add_argument('-d', '--dataset', type=str, default='mars',choices=datasets.names())  #s
    parser.add_argument('-b', '--batch-size', type=int, default=16)
    parser.add_argument('--epoch',type=int,default=30)
    parser.add_argument('--step_size',type=int,default=25)
    working_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--data_dir', type=str, metavar='PATH',default=os.path.join(working_dir, 'data'))  # 加载数据集的根目录
    parser.add_argument('--logs_dir', type=str, metavar='PATH',default=os.path.join(working_dir, 'logs'))  # 保持日志根目录
    parser.add_argument('--exp_name',type=str,default="gradully_supplement")
    parser.add_argument('--exp_order',type=str,default="1")
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--mode', type=str, choices=["Classification", "Dissimilarity"], default="Dissimilarity")   #这个考虑要不要取消掉
    parser.add_argument('--max_frames', type=int, default=100)
    parser.add_argument('--clock',type=bool, default=True)  #是否记时
    parser.add_argument('--gdraw',type=bool, default=False)  #是否实时绘图
    parser.add_argument('--total_step',type=int,default=5)  #默认总的五次迭代.
    # parser.add_argument('--')

    #下面是暂时不知道用来做什么的参数
    parser.add_argument('-a', '--arch', type=str, default='avg_pool',choices=models.names())  #eug model_name
    parser.add_argument('-i', '--iter-step', type=int, default=5)
    parser.add_argument('-g', '--gamma', type=float, default=0.3)
    parser.add_argument('-l', '--l', type=float)
    parser.add_argument('--continuous', action="store_true")
    main(parser.parse_args())
