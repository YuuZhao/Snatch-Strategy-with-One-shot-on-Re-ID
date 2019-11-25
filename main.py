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

'''start from checkpoint'''
def resume(args):
    import re
    pattern = re.compile(r'step_(\d+)\.ckpt')
    start_step = -1
    ckpt_file = ""

    # find start step
    files = os.listdir(osp.join("logs",args.dataset,args.exp_name,args.exp_order))
    files.sort()
    for filename in files:
        try:
            iter_ = int(pattern.search(filename).groups()[0])
            if iter_ > start_step:
                start_step = iter_
                ckpt_file = osp.join("logs",args.dataset,args.exp_name,args.exp_order,filename)
        except:
            continue

    # if need resume
    if start_step >= 0:
        print("continued from iter step", start_step)

    return start_step, ckpt_file


'''动态绘器'''
class gif_drawer():
    def __init__(self):
        plt.ion()
        self.select_num_percent = [0, 0]
        self.top1 = [0, 0]
        self.mAP = [0,0]
        self.label_pre = [0,0]
        self.select_pre = [0,1]
        self.flag = 0

    def draw(self, update_x, update_top1,mAP,label_pre,select_pre):
        self.select_num_percent[0] = self.select_num_percent[1]
        self.top1[0] = self.top1[1]
        self.mAP[0] = self.mAP[1]
        self.label_pre[0] = self.label_pre[1]
        self.select_pre[0] = self.select_pre[1]
        self.select_num_percent[1] = update_x
        self.top1[1] = update_top1
        # self.select_num_percent[1] = select_num_percent
        self.mAP[1] = mAP
        self.label_pre[1] = label_pre
        self.select_pre[1] = select_pre

        plt.title("Performance monitoring")
        plt.xlabel("select_percent(%)")
        plt.ylabel("value(%)")
        plt.plot(self.select_num_percent, self.top1, c="r", marker ='o',label="top1")
        plt.plot(self.select_num_percent, self.mAP, c="y", marker ='o',label="mAP")
        plt.plot(self.select_num_percent, self.label_pre, c="b", marker ='o',label="label_pre")
        plt.plot(self.select_num_percent, self.select_pre, c="cyan", marker ='o',label="select_pre")
        if self.flag==0:
            plt.legend()
            self.flag=1

    def saveimage(self,picture_path):
        plt.savefig(picture_path)

def changetoHSM(secends):
    m, s = divmod(secends, 60)
    h, m = divmod(m, 60)
    return h,m,s

def main(args):
    # 声明动态绘图器
    gd = gif_drawer()


    cudnn.benchmark = True
    cudnn.enabled = True

    # get all the labeled and unlabeled data for training
    dataset_all = datasets.create(args.dataset, osp.join(args.data_dir, args.dataset))
    l_data, u_data = get_one_shot_in_cam1(dataset_all, load_path="./examples/oneshot_{}_used_in_paper.pickle".format(
        dataset_all.name))
    NN = len(l_data) + len(u_data)

    # 总的训练step数的计算
    total_step = math.ceil(math.pow((100 / args.EF), (1 / args.q)))   # 这里应该取上限或者 +2  多一轮进行one-shot训练的  # EUG base 采样策略
    # total_step = math.ceil((2 * NN * args.step_s + args.yita + len(u_data)) / (args.yita + NN + len(l_data))) + 2 # big start 策略

    # 输出该轮训练关键的提示信息
    print("{} training begin with dataset:{},batch_size:{},epoch:{},step_size:{},max_frames:{},total_step:{},EF:{},q:{},yita:{},step_s:{}".format(args.exp_name,args.dataset,args.batch_size,args.epoch,args.step_size,args.max_frames,total_step+1,args.EF,args.q,args.yita,args.step_s))

    # 指定输出文件
    # 第三部分要说明关键参数的设定
    sys.stdout = Logger(osp.join(args.logs_dir, args.dataset,args.exp_name,args.exp_order,'log'+time.strftime(".%m_%d_%H-%M-%S")+'.txt'))
    data_file =codecs.open(osp.join(args.logs_dir, args.dataset,args.exp_name,args.exp_order,'data.txt'),mode='a')
    if args.clock :
        time_file =codecs.open(osp.join(args.logs_dir, args.dataset,args.exp_name,args.exp_order,'time.txt'),mode='a')
    save_path = osp.join(args.logs_dir, args.dataset,args.exp_name,args.exp_order)

    resume_step, ckpt_file = -1, ''
    if args.resume:  # 重新训练的时候用
        resume_step, ckpt_file = resume(args)

    # initial the EUG algorithm
    eug = EUG(model_name=args.arch, batch_size=args.batch_size, mode=args.mode, num_classes=dataset_all.num_train_ids,
              data_dir=dataset_all.images_dir, l_data=l_data, u_data=u_data, save_path=save_path,
              max_frames=args.max_frames)

    # 训练之前初始化数据
    nums_to_select = 0
    new_train_data = l_data
    step = 0
    step_size = []
    isout = 0  #用来标记是否应该结束训练

    # 开始的时间记录
    exp_start = time.time()
    while(not isout):
        print("{} training begin with dataset:{},batch_size:{},epoch:{},step:{}/{} saved to {}.".format(args.exp_name,args.dataset,args.batch_size, args.epoch,step+1,total_step+1,save_path))
        print("key parameters contain EF:{},q:{},yita:{},step_s:{}. Nums_been_selected:{}".format(args.EF,args.q,args.yita,args.step_s,nums_to_select))

        # 开始训练
        train_start = time.time()
        eug.train(new_train_data, step, epochs=args.epoch, step_size=args.step_size, init_lr=0.1) if step != resume_step else eug.resume(ckpt_file, step)

        # 开始评估
        evaluate_start = time.time()
        # mAP, top1, top5, top10, top20 = 0,0,0,0,0
        mAP,top1,top5,top10,top20 = eug.evaluate(dataset_all.query, dataset_all.gallery)

        # 标签估计
        estimate_start = time.time()
        # pred_y, pred_score, label_pre, id_num = 0,0,0,0
        pred_y, pred_score, label_pre, id_num = eug.estimate_label()
        estimate_end = time.time()

        # 循环退出判断
        if nums_to_select == len(u_data):
            isout = 1

        # nums_to_select 的设定
        new_nums_to_select = min(math.ceil(len(u_data) * math.pow((step + 1), args.q) * args.EF / 100),len(u_data))  # EUG 基础指数渐进策略
        # new_nums_to_select = min(math.ceil((len(u_data)-args.yita)*(step-1)/(total_step-2))+args.yita,len(u_data))  # big start

        selected_idx = eug.select_top_data(pred_score, new_nums_to_select)
        new_train_data, select_pre = eug.generate_new_train_data(selected_idx, pred_y)

        # 输出该epoch的信息
        data_file.write("step:{} mAP:{:.2%} top1:{:.2%} top5:{:.2%} top10:{:.2%} top20:{:.2%} nums_selected:{} selected_percent:{:.2%} label_pre:{:.2%} select_pre:{:.2%}\n".format(
                int(step+1), mAP, top1, top5,top10,top20,nums_to_select, nums_to_select/len(u_data),label_pre,select_pre))
        print(
            "step:{} mAP:{:.2%} top1:{:.2%} top5:{:.2%} top10:{:.2%} top20:{:.2%} nums_selected:{} selected_percent:{:.2%} label_pre:{:.2%} select_pre:{:.2%}\n".format(
                int(step+1), mAP, top1, top5, top10, top20, nums_to_select, nums_to_select / len(u_data), label_pre,select_pre))

        if args.clock:
            train_time = evaluate_start-train_start
            evaluate_time = estimate_start - evaluate_start
            estimate_time = estimate_end-estimate_start
            epoch_time = train_time+estimate_time
            time_file.write("step:{}  train:{} evaluate:{} estimate:{} epoch:{}\n".format(int(step+1),train_time,evaluate_time,estimate_time,epoch_time))

        if args.gdraw:
            gd.draw(nums_to_select/len(u_data),top1,mAP,label_pre,select_pre)

        nums_to_select = new_nums_to_select
        step = step + 1

    data_file.close()
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
    parser.add_argument('--EF', type=float, default=5)  # 渐进采样系数
    parser.add_argument('--q', type=float, default=1)  # 渐进采样指数
    parser.add_argument('--yita', type=int, default=100)   #big start based number
    parser.add_argument('--step_s', type=int, default=10)  #big start 饱和度控制
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

    #下面是暂时不知道用来做什么的参数
    parser.add_argument('-a', '--arch', type=str, default='avg_pool',choices=models.names())  #eug model_name
    parser.add_argument('-i', '--iter-step', type=int, default=5)
    parser.add_argument('-g', '--gamma', type=float, default=0.3)
    parser.add_argument('-l', '--l', type=float)
    parser.add_argument('--continuous', action="store_true")
    main(parser.parse_args())
