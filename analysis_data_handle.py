import argparse
import os.path as osp
import os
import codecs
# import mydatabase
import math
import matplotlib.pyplot as plt
import numpy as np
import json
# from data_draw import summary_gradually_compare
font2={
        'family':'DejaVu Sans',#'Times New Roman',
        'weight':'normal',
        'size':22,
    }

'''
    文件描述 :
    this file is used to draw figure according  to the data produced by 'analysis.py'.
    you must give the special parameters 'exp_name' and 'exp_order' which indicate the location of the data to run this files.
    运行命令如:
    python3.6  analysis_data_handle.py --exp_name gradually_11step --exp_order 0
'''

def main(args):
    input_path = osp.join(args.logs_dir, args.dataset, args.exp_name, args.exp_order, 'analysis')
    dists_path = osp.join(input_path, 'dists')
    vari_path = osp.join(input_path, 'vari')
    acc_path = osp.join(input_path, 'acc_list')
    if args.function==1:
        '''
        draw figures for every item in 'analysis' folder.
        in each item, compare data from different step of model.
        > dists:
            sort it from little to big before drew picture.
        > vari:
            sort list from  little to big.
        '''

        plt.figure(figsize=(12,6),dpi=300)
        #dists_files = os.listdir(dists_path)
        #dists_files.sort() #this method can not perform as expectation fully.
        for i in range(11):
            dists_file = 'dists{}.npy'.format(i)
            dists = np.load(osp.join(dists_path,dists_file))
            x = np.arange(dists.shape[0])
            dists.sort()
            plt.plot(x,dists,label='model_{}'.format(i))
        plt.legend(loc='best')
        plt.title('sorted distance')
        plt.xlabel('u_data')
        plt.ylabel('distance')
        plt.savefig(osp.join(input_path,'dists_analysis_{}'.format(args.function)))
        plt.show()

        plt.figure(figsize=(12, 6), dpi=300)
        #vari_files = os.listdir(vari_path)
        ##vari_files.sort()
        for i in range(11):
            vari_file = 'vari{}.npy'.format(i)
            vari = np.load(osp.join(vari_path, vari_file))
            x = np.arange(vari.shape[0])
            vari = -np.sort(-vari)
            plt.plot(x,vari, label='model_{}'.format(i))
        plt.legend(loc='best')
        plt.title('sorted variance')
        plt.xlabel('u_data')
        plt.ylabel('variance')
        plt.savefig(osp.join(input_path, 'vari_analysis_{}'.format(args.function)))
        plt.show()

    elif args.function==2:
        '''
        this part aims to draw the relationship between 'acc_list'  and both  'dists' and 'vari'
        specially, we first sort index according to dists or vari, than draw the change of acc_list.
        '''
        print('excute function 2')
        plt.figure(figsize=(12, 6), dpi=300)
        for i in range(11):
            #plt.subplot(10, 1, step + 1)
            #print('handle dists{}.npy'.format(i))
            dists_file = 'dists{}.npy'.format(i)
            dists = np.load(osp.join(dists_path,dists_file))
            acc_file = 'acc_list{}.npy'.format(i)
            acc_list = np.load(osp.join(acc_path,acc_file))
            list_length = dists.shape[0]
            x = np.arange(list_length)
            index = np.argsort(dists)
            y = acc_list[index]
            acc = np.zeros(list_length)
            for j in range(list_length):  # 依次求acc
                acc[j] = y[0:j + 1].sum() / (j + 1)
            plt.plot(x, acc, label='model_{}'.format(i))
            #z = np.zeros(dists.shape[0]) + i
            #plt.bar(x, y,bottom=z, label= dists_file.split('.')[0]) # it is always wrong
        plt.legend(loc='best')
        plt.title('FN_acc sort by distance')
        plt.xlabel('u_data')
        plt.ylabel('FN_acc')
        plt.savefig(osp.join(input_path,'dists_analysis_{}'.format(args.function)))
        plt.show()

        plt.figure(figsize=(12, 6), dpi=300)
        for i in range(11):
            vari_file = 'vari{}.npy'.format(i)
            vari = np.load(osp.join(vari_path, vari_file))
            acc_file = 'acc_list{}.npy'.format(i)
            acc_list = np.load(osp.join(acc_path, acc_file))
            list_length = vari.shape[0]
            x = np.arange(list_length)
            index = np.argsort(-vari)
            y = acc_list[index]
            acc = np.zeros(list_length)
            for j in range(list_length):  # 依次求acc
                acc[j] = y[0:j + 1].sum() / (j + 1)
            plt.plot(x, acc, label='model_{}'.format(i))
            #z = np.zeros(vari.shape[0])+i
            #plt.bar(x, y, bottom=z, label=vari_file.split('.')[0])
        plt.legend(loc='best')
        plt.title('FN_acc sort by variance')
        plt.xlabel('u_data')
        plt.ylabel('FN_acc')
        plt.savefig(osp.join(input_path, 'vari_analysis_{}'.format(args.function)))
        plt.show()

    elif args.function==3:
        '''
        sorted dists up first. then report the vari responsiblely.
        '''
        print('excute function 3')
        plt.figure(figsize=(12, 6), dpi=300)
        for i in range(11):
            dists_file = 'dists{}.npy'.format(i)
            dists = np.load(osp.join(dists_path, dists_file))
            #acc_file = 'acc_list{}.npy'.format(i)
            #acc_list = np.load(osp.join(acc_path, acc_file))
            vari_file = 'vari{}.npy'.format(i)
            vari = np.load(osp.join(vari_path,vari_file))
            list_length = dists.shape[0]
            x = np.arange(list_length)
            index = np.argsort(dists)
            y = vari[index]
            plt.plot(x, y, label='model_{}'.format(i))
        plt.legend(loc='best')
        plt.title('variance sort by distance')
        plt.xlabel('u_data')
        plt.ylabel('variance')
        plt.savefig(osp.join(input_path, 'dists&vari_1_analysis_{}'.format(args.function)))
        plt.show()

        plt.figure(figsize=(12, 6), dpi=300)
        plt.suptitle('variance sort by distance')
        plt.xlabel('u_data')
        plt.ylabel('variance')
        for i in range(11):
            plt.subplot(3,4,i+1)
            dists_file = 'dists{}.npy'.format(i)
            dists = np.load(osp.join(dists_path, dists_file))
            # acc_file = 'acc_list{}.npy'.format(i)
            # acc_list = np.load(osp.join(acc_path, acc_file))
            vari_file = 'vari{}.npy'.format(i)
            vari = np.load(osp.join(vari_path, vari_file))
            list_length = dists.shape[0]
            x = np.arange(list_length)
            index = np.argsort(dists)
            y = vari[index]
            plt.plot(x, y, label='model_{}'.format(i))
            plt.legend(loc='best')
            #plt.xlabel('u_data')
            #plt.ylabel('variance')
        plt.savefig(osp.join(input_path, 'dists&vari_11_analysis_{}'.format(args.function)))
        plt.show()

    elif args.function == 4:
        '''
        compare model1 and model2
        sepcially, get the distance sorted index of model1, then draw the distance fo model2 according to the index.
        '''
        plt.figure(figsize=(12, 6), dpi=300)
        plt.title('distance change of models')
        plt.xlabel('u_data')
        plt.ylabel('distance')
        dists1 = np.load(osp.join(dists_path, 'dists1.npy'))
        dists2 = np.load(osp.join(dists_path, 'dists2.npy'))
        dists3 = np.load(osp.join(dists_path, 'dists3.npy'))
        list_length = dists2.shape[0]
        x = np.arange(list_length)
        index1 = np.argsort(dists1)
        index2 = np.argsort(dists2)
        y1 = dists1[index1]
        y2 = dists2[index1]
        y3 = dists2[index2]
        y4 = dists3[index2]
        plt.plot(x, y4, label='dists3 sorted by dists2')
        plt.plot(x, y2, label='dists2 sorted by dists1')
        plt.plot(x, y1, label='dists1')
        plt.plot(x, y3, label='dists2')
        plt.legend(loc='best')
        plt.savefig(osp.join(input_path, 'dists_analysis_{}'.format(args.function)))
        plt.show()


    elif args.function == 5:
        '''
        analyze the relationship of dists,vari and acc_list.
        take  the dists x axis, draw the vari and FN_acc curves sorted by dists up.
        '''
        plt.figure(figsize=(24, 12), dpi=300)
        plt.suptitle('relationship between variance, dists, acc_list[{}:{}]'.format(args.Start_point,args.End_point),fontsize=22)
        plt.xlabel('distance')
        plt.ylabel('variance')
        for i in range(11):
            ax2 =plt.subplot(3, 4, i + 1)
            dists_file = 'dists{}.npy'.format(i)
            dists = np.load(osp.join(dists_path, dists_file))
            acc_file = 'acc_list{}.npy'.format(i)
            acc_list = np.load(osp.join(acc_path, acc_file))
            vari_file = 'vari{}.npy'.format(i)
            vari = np.load(osp.join(vari_path, vari_file))
            index = np.argsort(dists)
            dists = dists[index]
            vari = vari[index]
            list_length = dists.shape[0]
            x = np.arange(list_length)
            acc_list = acc_list[index]
            acc = np.zeros(list_length)
            for j in range(list_length):  # 依次求acc
                acc[j] = acc_list[0:j + 1].sum() / (j + 1)

            lns2 = ax2.plot(x[args.Start_point:args.End_point], acc[args.Start_point:args.End_point], label='model_{}_FNacc'.format(i))
            lns3 = ax2.plot(x[args.Start_point:args.End_point], acc_list[args.Start_point:args.End_point] *0.2,'bo',ms=1,color='g', label='model_{}_acc_list'.format(i))
            ax1 = ax2.twinx()
            lns1 = ax1.plot(x[args.Start_point:args.End_point], vari[args.Start_point:args.End_point], color='coral', label='model_{}_vari'.format(i))
            lns =lns1+lns2+lns3
            labs = [l.get_label() for l in lns]
            if i==10:
                ax1.legend(lns,labs, loc='lower right',bbox_to_anchor=(2,0))
            ax1.set_xlabel('distance')
            ax1.set_ylabel('variance')
            ax2.set_ylabel('FN_acc')
        plt.savefig(osp.join(input_path, 'dists&vari&FNacc[{}:{}]_11_analysis_{}'.format(args.Start_point,args.End_point,args.function)))
        #plt.show()


















if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='analysis_data_handle')
    parser.add_argument('--function',type=int,default=1)
    parser.add_argument('--End_point',type=int,default=-1)
    parser.add_argument('--Start_point',type=int,default=0)
    working_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--data_dir', type=str, metavar='PATH', default=os.path.join(working_dir, 'data'))  # 加载数据集的根目录
    parser.add_argument('--logs_dir', type=str, metavar='PATH', default=os.path.join(working_dir, 'logs'))  # 保持日志根目录
    parser.add_argument('--exp_name', type=str, default="epsm")
    parser.add_argument('--exp_order', type=str, default="0")
    parser.add_argument('--dataset', type=str, default='DukeMTMC-VideoReID')
    main(parser.parse_args())