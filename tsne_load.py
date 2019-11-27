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
    # gd = gif_drawer2()

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("game begin!")
    cudnn.benchmark = True
    cudnn.enabled = True
    save_path = args.logs_dir
    total_step = math.ceil(math.pow((100 / args.EF), (1 / args.q))) + 1  # 这里应该取上限或者 +2  多一轮进行one-shot训练的
    print("total_step:{}".format(total_step))
    sys.stdout = Logger(osp.join(args.logs_dir, 'log' + str(args.EF)+"_"+ str(args.q) + time.strftime(".%m_%d_%H-%M-%S") + '.txt'))
    data_file =codecs.open(osp.join(args.logs_dir,'data' + str(args.EF)+"_"+ str(args.q) + time.strftime(".%m_%d_%H-%M-%S") + '.txt'),'a')
    time_file =codecs.open(osp.join(args.logs_dir,'time' + str(args.EF)+"_"+ str(args.q) + time.strftime(".%m_%d_%H-%M-%S") + '.txt'),'a')

    # get all the labeled and unlabeled data for training
    dataset_all = datasets.create(args.dataset, osp.join(args.data_dir, args.dataset))
    num_all_examples = len(dataset_all.train)
    l_data, u_data = get_one_shot_in_cam2(dataset_all, load_path="./examples/oneshot_{}_used_in_paper.pickle".format(
        dataset_all.name))

        # initial the EUG algorithm
    eug = EUG(model_name=args.arch, batch_size=1024, mode=args.mode, num_classes=dataset_all.num_train_ids,
              data_dir=dataset_all.images_dir, l_data=l_data, u_data=u_data, save_path=args.logs_dir,
              max_frames=args.max_frames)
    middle = math.ceil(len(l_data)/2)
    l_data_1 = l_data[:middle]
    l_data_2 = l_data[middle:]
    eug.resume('tsne/Dissimilarity_step_0.ckpt', 0)
    print('Extracting features...')
    fts,lbs,cams = eug.get_feature_with_labels_cams(l_data_1)
    print('Saving fts1...')
    np.save('tsne/mars/Dissimilarity_step_0_fts_1.npy',fts)
    print('Saving lbs1...')
    np.save('tsne/mars/Dissimilarity_step_0_lbs_1.npy',lbs)
    print('Saving cams1...')
    np.save('tsne/mars/Dissimilarity_step_0_cams_1.npy',cams)
    fts, lbs, cams = eug.get_feature_with_labels_cams(l_data_2)
    print('Saving fts1...')
    np.save('tsne/mars/Dissimilarity_step_0_fts_2.npy', fts)
    print('Saving lbs1...')
    np.save('tsne/mars/Dissimilarity_step_0_lbs_2.npy', lbs)
    print('Saving cams1...')
    np.save('tsne/mars/Dissimilarity_step_0_cams_2.npy', cams)
    print('Done.')






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Exploit the Unknown Gradually')
    parser.add_argument('-d', '--dataset', type=str, default='mars',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=1024)
    parser.add_argument('-a', '--arch', type=str, default='avg_pool',
                        choices=models.names())
    parser.add_argument('-i', '--iter-step', type=int, default=5)
    parser.add_argument('-g', '--gamma', type=float, default=0.3)
    parser.add_argument('-l', '--l', type=float)
    parser.add_argument('--EF', type=float, default=5)
    parser.add_argument('--q', type=float, default=1)  # 指数参数
    parser.add_argument('--k', type=float, default=15)
    parser.add_argument('--bs', type=int, default=50)
    working_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--data_dir', type=str, metavar='PATH',
                        default=os.path.join(working_dir, 'data'))
    parser.add_argument('--logs_dir', type=str, metavar='PATH',
                        default='logs/mars/')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--continuous', action="store_true")
    parser.add_argument('--mode', type=str, choices=["Classification", "Dissimilarity"], default="Dissimilarity")
    parser.add_argument('--max_frames', type=int, default=100)
    main(parser.parse_args())
