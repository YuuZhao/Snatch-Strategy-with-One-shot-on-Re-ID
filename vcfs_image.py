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



def main(args):
    input_path = osp.join(args.logs_dir, args.dataset, args.exp_name, args.exp_order, 'analysis')
    dists_path = osp.join(input_path, 'dists')
    vari_path = osp.join(input_path, 'vari')
    acc_path = osp.join(input_path, 'acc_list')

    def function7(args,acc_file_name):
        plt.figure(figsize=(12, 6), dpi=300)
        plt.suptitle(
            'relationship between variance, dists, acc_list[{}:{}] of model_{}'.format(args.start_point, args.end_point,
                                                                                       args.model_order),
            fontsize=18)
        # plt.xlabel('distance')
        # plt.ylabel('variance')
        for i in [args.model_order]:

            ax2 = plt.subplot(1, 1, 1)
            dists_file = 'dists{}.npy'.format(i)
            dists = np.load(osp.join(dists_path, dists_file))
            list_length = dists.shape[0]
            acc_file = 'acc_list{}.npy'.format(i)
            acc_list = np.load(osp.join(acc_path, acc_file))
            vari_file = 'vari{}.npy'.format(i)
            vari = np.load(osp.join(vari_path, vari_file))

            expand_to_num = math.floor(args.end_point * args.expand_rate)
            query_start_num = math.floor(args.end_point * args.query_rate)
            if expand_to_num >= list_length:
                expand_to_num = list_length - 1

            index1 = np.argsort(dists)
            vari_sort_distance = vari[index1]

            x = np.arange(list_length)
            acc_sort_distance = acc_list[index1]
            FNacc1 = np.zeros(list_length)
            for j in range(list_length):  # 依次求acc
                FNacc1[j] = acc_sort_distance[0:j + 1].sum() / (j + 1)
            lns2 = ax2.plot(x[args.start_point:expand_to_num], FNacc1[args.start_point:expand_to_num],
                            label='model_{}_FNacc'.format(i))
            lns3 = ax2.plot(x[args.start_point:expand_to_num],
                            acc_sort_distance[args.start_point:expand_to_num] * 0.2,
                            'bo',
                            ms=1, color='g', label='model_{}_acc_list'.format(i))
            ax1 = ax2.twinx()
            lns1 = ax1.plot(x[args.start_point:expand_to_num], vari_sort_distance[args.start_point:expand_to_num],
                            color='coral',
                            label='model_{}_vari'.format(i))

            # resort by variance in [query_start_num:expand_to_num]
            acc1 = acc_sort_distance[:args.end_point].sum()  #
            vari_range = vari_sort_distance[:expand_to_num]  #
            index2 = np.argsort(-vari_range)  # down sort
            second_sampling_index2 = index2[:args.end_point]
            acc_sort_variance = acc_sort_distance[second_sampling_index2]
            acc2 = acc_sort_variance.sum()  # /second_samping_num
            #vari_range2 = vari_sort_distance[query_start_num:expand_to_num]
            index3 = np.argsort(-vari_sort_distance[query_start_num:expand_to_num]) + query_start_num
            second_sampling_index3 = np.append(np.arange(query_start_num),
                                               index3[:args.end_point - query_start_num])
            acc_sort_variance3 = acc_sort_distance[second_sampling_index3]
            acc3 = acc_sort_variance3.sum()
            print('second_sampling_index3\'s length ={}'.format(len(second_sampling_index3)))
            print(
                'model_{} :: acc1:{}/{:.2%}, acc2:{}/{:.2%} acc3:{}/{:.2%}'.format(i, acc1, acc1 / args.end_point,
                                                                                   acc2,
                                                                                   acc2 / args.end_point, acc3,
                                                                                   acc3 / args.end_point))
            acc_file = codecs.open(acc_file_name,mode='a')
            acc_file.write('{}/{:.2%},{}/{:.2%},{}/{:.2%}\n'.format(acc1,acc1 / args.end_point,
                                                                                   acc2,
                                                                                   acc2 / args.end_point, acc3,
                                                                                   acc3 / args.end_point))


            vari_flag = vari_sort_distance[:expand_to_num]
            for j in range(expand_to_num):
                if j not in second_sampling_index3:
                    # print("not in index2")
                    vari_flag[j] = 0.3
            lns4 = ax1.plot(x[args.start_point:expand_to_num], vari_flag, 'bo', ms=1,
                            label='model_{}_vari400'.format(i),
                            color='r')
            lns = lns1 + lns2 + lns3 + lns4
            labs = [l.get_label() for l in lns]
            ax1.legend(lns, labs, loc='best')
            # if i == 10:
            #     ax1.legend(lns, labs, loc='lower right', bbox_to_anchor=(2, 0))
            ax1.set_xlabel('distance')
            ax1.set_ylabel('variance')
            ax2.set_ylabel('FN_acc')
        plt.savefig(osp.join(input_path,
                             'dists&vari&FNacc[{}:{}]_model_{}_analysis_{}'.format(args.start_point, args.end_point,
                                                                                   args.model_order,
                                                                                   args.function)))
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
        for i in range(8):
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
        for i in range(8):
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
        for i in range(8):
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
        for i in range(8):
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
        for i in range(8):
            dists_file = 'dists{}.npy'.format(i)
            dists = np.load(osp.join(dists_path, dists_file))
            #acc_file = 'acc_list{}.npy'.format(i)
            #acc_list = np.load(osp.join(acc_path, acc_file))
            vari_file = 'vari{}.npy'.format(i)
            vari = np.load(osp.join(vari_path,vari_file))
            list_length = dists.shape[0]
            x = np.arange(list_length)
            # index = np.argsort(dists)
            index = np.argsort(-vari)
            y = dists[index]
            plt.plot(x, y, label='model_{}'.format(i))
        plt.legend(loc='best')
        plt.title('distance sort by variance')
        plt.xlabel('u_data')
        plt.ylabel('distance')
        plt.savefig(osp.join(input_path, 'dists&vari_1_analysis_{}'.format(args.function)))
        plt.show()

        plt.figure(figsize=(12, 6), dpi=300)
        plt.suptitle('variance sort by distance')
        plt.xlabel('u_data')
        plt.ylabel('variance')
        for i in range(8):
            plt.subplot(3,3,i+1)
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


    elif args.function == 5:  #not run again
        '''
        analyze the relationship of dists,vari and acc_list.
        take  the dists x axis, draw the vari and FN_acc curves sorted by dists up.
        '''
        plt.figure(figsize=(24, 12), dpi=300)
        plt.suptitle('relationship between variance, dists, acc_list[{}:{}]'.format(args.start_point,args.end_point),fontsize=22)
        plt.xlabel('distance')
        plt.ylabel('variance')
        for i in range(8):
            ax2 =plt.subplot(3, 3, i + 1)
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

            lns2 = ax2.plot(x[args.start_point:args.end_point], acc[args.start_point:args.end_point], label='model_{}_FNacc'.format(i))
            lns3 = ax2.plot(x[args.start_point:args.end_point], acc_list[args.start_point:args.end_point] *0.2,'bo',ms=1,color='g', label='model_{}_acc_list'.format(i))
            ax1 = ax2.twinx()
            lns1 = ax1.plot(x[args.start_point:args.end_point], vari[args.start_point:args.end_point], color='coral', label='model_{}_vari'.format(i))
            lns =lns1+lns2+lns3
            labs = [l.get_label() for l in lns]
            if i==7:
                ax1.legend(lns,labs, loc='lower right',bbox_to_anchor=(2,0))
            ax1.set_xlabel('distance')
            ax1.set_ylabel('variance')
            ax2.set_ylabel('FN_acc')
        plt.savefig(osp.join(input_path, 'dists&vari&FNacc[{}:{}]_11_analysis_{}'.format(args.start_point,args.end_point,args.function)))
        #plt.show()

    elif args.function == 6: #jump out
        '''
        analyze the relationship of dists,vari and acc_list. (variance second sampling)
        take  the dists x axis, draw the vari and FN_acc curves sorted by dists up.
        '''
        plt.figure(figsize=(12, 6), dpi=300)
        plt.suptitle('relationship between variance, dists, acc_list[{}:{}]'.format(args.start_point, args.end_point),
                     fontsize=18)
        plt.xlabel('distance')
        plt.ylabel('variance')
        for i in [0]:
            ax2 = plt.subplot(1, 1, 1)
            dists_file = 'dists{}.npy'.format(i)
            dists = np.load(osp.join(dists_path, dists_file))
            list_length = dists.shape[0]
            acc_file = 'acc_list{}.npy'.format(i)
            acc_list = np.load(osp.join(acc_path, acc_file))
            vari_file = 'vari{}.npy'.format(i)
            vari = np.load(osp.join(vari_path, vari_file))
            index1 = np.argsort(dists)
            vari_sort_distance = vari[index1]

            x = np.arange(list_length)
            acc_sort_distance = acc_list[index1]
            FNacc1 = np.zeros(list_length)
            for j in range(list_length):  # 依次求acc
                FNacc1[j] = acc_sort_distance[0:j + 1].sum() / (j + 1)
            lns2 = ax2.plot(x[args.start_point:args.end_point], FNacc1[args.start_point:args.end_point],
                            label='model_{}_FNacc'.format(i))
            lns3 = ax2.plot(x[args.start_point:args.end_point], acc_sort_distance[args.start_point:args.end_point] * 0.2, 'bo',
                            ms=1, color='g', label='model_{}_acc_list'.format(i))
            ax1 = ax2.twinx()
            lns1 = ax1.plot(x[args.start_point:args.end_point], vari_sort_distance[args.start_point:args.end_point], color='coral',
                            label='model_{}_vari'.format(i))

            second_samping_num = math.floor(args.end_point* args.second_sampling_rate)

            #first_sampling_index = index1[:second_samping_num]  # the first 200 samples are selected.
            acc1 = acc_sort_distance[:second_samping_num].sum()#/second_samping_num
            #print("length of acc1= {}".format(len(acc_list[first_sampling_index])))
            vari_range = vari_sort_distance[:args.end_point]  #
            index2 = np.argsort(-vari_range)  # down sort
            second_sampling_index = index2[:second_samping_num]
            #print("second_sampling_index length ={}".format(len(second_sampling_index)))
            acc2 = acc_sort_distance[second_sampling_index].sum()#/second_samping_num
            print("length of acc2= {}".format(len(acc_list[second_sampling_index])))
            print('model_{} :: acc1:{}, acc2:{} up {}'.format(i,acc1,acc2,(acc2-acc1)/acc1))
            vari_flag = vari_sort_distance[:args.end_point]
            for j in range(args.end_point):
                if j not in index2[:second_samping_num]:
                    #print("not in index2")
                    vari_flag[j] = 0.3

            lns4 = ax1.plot(x[args.start_point:args.end_point],vari_flag,'bo',ms=1,label='model_{}_vari400'.format(i),color='r')
            lns = lns1 + lns2 + lns3 +lns4
            labs = [l.get_label() for l in lns]
            ax1.legend(lns,labs,loc='best')
            # if i == 10:
            #     ax1.legend(lns, labs, loc='lower right', bbox_to_anchor=(2, 0))
            ax1.set_xlabel('distance')
            ax1.set_ylabel('variance')
            ax2.set_ylabel('FN_acc')
        plt.savefig(osp.join(input_path,
                             'dists&vari&FNacc[{}:{}]_11_analysis_{}'.format(args.start_point, args.end_point,
                                                                             args.function)))



    elif args.function == 7: #jump out
        '''
        analyze the relationship of dists,vari and acc_list. (variance second sampling)
        take  the dists x axis, draw the vari and FN_acc curves sorted by dists up.
        add query_rate based on funciton 6.
        '''
        function7(args)


    elif args.function ==8:
        '''
        varify the accurcy of dists, varice, query_varian
        aim to get acc information for model0-model10
        '''
        def sampling_curve(x): # x in [0,10]
            return math.floor( 1494* x /8)  #7673

        # expand_rate = [1.0,1.02,1.04,1.06,1.08,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]
        # query_rate = [0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        expand_rate = [1.1]
        query_rate = [0.8]

        for er in expand_rate:
            acc_file_name = osp.join(input_path, 'acc_file_[0.8:{}].txt'.format(er))
            args.expand_rate = er
            args.query_rate = 0.8
            for i in range(8):
                args.model_order = i
                args.end_point = sampling_curve(i+1)
                function7(args, acc_file_name)
        # for qr in query_rate:
        #     acc_file_name = osp.join(input_path, 'acc_file_[{}:1.1].txt'.format(qr))
        #     args.expand_rate = 1.1
        #     args.query_rate = qr
        #     for i in range(8):
        #         args.model_order = i
        #         args.end_point = sampling_curve(i + 1)
        #         function7(args, acc_file_name)




    elif args.function==9:
        '''
        draw acc_file curve according to txt-data in acc_file folder.
        '''
        acc_file_path = osp.join(input_path,'acc_file')
        x = [1.0,1.02,1.04,1.06,1.08,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]
        rate_list =[]
        for expand_rate in x:
            acc_file = codecs.open(osp.join(acc_file_path,'acc_file_[0.8:{}].txt'.format(expand_rate)),'r','utf-8')
            lines = acc_file.readlines()
            model_list = []
            for line in lines:
                acc_value = []
                information = line.strip().split(',')
                for acc_info in information:
                    value = int(acc_info.split('/')[0].strip('.0'))
                    percent = float(acc_info.split('/')[1].strip('%'))
                    acc_value.extend([value,percent])
                model_list.append(acc_value)
            rate_list.append(model_list)
        # until there ,all the data hava been extracted.
        rate_list = np.array(rate_list)
        plt.figure(figsize=(20,12), dpi=300)
        # plt.xlabel('expand_rate')
        # plt.ylabel('accuracy_percent')
        for model_order in range(8):
            plt.subplot(3,3,model_order+1)
            acc1_percent = rate_list[:,model_order,1]
            acc1_num = rate_list[:,model_order,0]
            acc2_percent = rate_list[:,model_order, 3]
            acc2_num = rate_list[:,model_order, 2]
            acc3_percent = rate_list[:,model_order, 5]
            acc3_num = rate_list[:,model_order, 4]
            max_point = np.argmax(acc3_percent)
            lacal_x = x[max_point]
            plt.annotate(str([lacal_x,acc3_percent[max_point]]), xy=(lacal_x, acc3_percent[max_point]), fontsize=15)
            plt.plot(x,acc1_percent,label='acc1_percent')
            plt.plot(x,acc2_percent,label='acc2_percent')
            plt.plot(x,acc3_percent,label='acc3_percent')
            #plt.yticks(np.arange(50,100,10))
            #plt.ylim((50,100))
            plt.xlabel('model_{}'.format(model_order))
            plt.legend(loc='best')
        plt.savefig(osp.join(input_path,'analysisi_{}'.format(args.function)))

    elif args.function==10:
        '''
        draw acc_file curve according to txt-data in acc_file folder.
        '''
        acc_file_path = osp.join(input_path,'acc_file')
        x = [0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        rate_list =[]
        for expand_rate in x:
            acc_file = codecs.open(osp.join(acc_file_path,'acc_file_[{}:1.1].txt'.format(expand_rate)),'r','utf-8')
            lines = acc_file.readlines()
            model_list = []
            for line in lines:
                acc_value = []
                information = line.strip().split(',')
                for acc_info in information:
                    value = int(acc_info.split('/')[0].strip('.0'))
                    percent = float(acc_info.split('/')[1].strip('%'))
                    acc_value.extend([value,percent])
                model_list.append(acc_value)
            rate_list.append(model_list)
        # until there ,all the data hava been extracted.
        rate_list = np.array(rate_list)
        plt.figure(figsize=(20,12), dpi=300)
        # plt.xlabel('expand_rate')
        # plt.ylabel('accuracy_percent')
        for model_order in range(8):
            plt.subplot(3,3,model_order+1)
            acc1_percent = rate_list[:,model_order,1]
            acc1_num = rate_list[:,model_order,0]
            acc2_percent = rate_list[:,model_order, 3]
            acc2_num = rate_list[:,model_order, 2]
            acc3_percent = rate_list[:,model_order, 5]
            acc3_num = rate_list[:,model_order, 4]
            max_point = np.argmax(acc3_percent)
            lacal_x = x[max_point]
            plt.annotate(str([lacal_x,acc3_percent[max_point]]), xy=(lacal_x, acc3_percent[max_point]), fontsize=15)
            plt.plot(x,acc1_percent,label='acc1_percent')
            plt.plot(x,acc2_percent,label='acc2_percent')
            plt.plot(x,acc3_percent,label='acc3_percent')
            # plt.yticks(np.arange(50,100,10))
            #plt.ylim((50,100))
            plt.xlabel('model_{}'.format(model_order))
            plt.legend(loc='best')
        plt.savefig(osp.join(input_path,'analysisi_{}'.format(args.function)))

    elif args.function == 11: # based on function 8
        query_rate_list = []
        expand_rate_list = []
        Nu = 0
        key_name = ''
        if args.dataset == 'mars':
            query_rate_list=[0.4,0.5,0.6,0.7,0.8,0.9,1,1]
            expand_rate_list = [1.2,1.1,1.08,1.06,1.04,1.02,1,1]
            Nu = 7673
        else:
            query_rate_list=[0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
            expand_rate_list = [1.3,1.2, 1.1, 1.08, 1.06, 1.04, 1.02, 1]
            Nu=1494
        if args.is_best_para ==1:
            key_name = 'best_parameter'
        else:
            key_name = 'q_half_e'

        def sampling_curve(x): # x in [0,10]
            return math.floor( Nu * x / len(query_rate_list))  #7673

        acc_file_name = osp.join(input_path, '{}_{}.txt'.format(key_name,args.function))
        plt.figure(figsize=(20, 14), dpi=300)
        plt.suptitle(
            '{}_of EF15_{}'.format(key_name,args.function),fontsize=18)
        for i in range(len(query_rate_list)): # huizhi model ge subfigure
            args.end_point = sampling_curve(i+1)
            args.expand_rate = expand_rate_list[i]
            if args.is_best_para ==1:
                args.query_rate = query_rate_list[i]
            else:
                args.query_rate = args.expand_rate / 2
            ax2 = plt.subplot(3, 3, i+1)
            dists_file = 'dists{}.npy'.format(i)
            dists = np.load(osp.join(dists_path, dists_file))
            list_length = dists.shape[0]
            acc_file = 'acc_list{}.npy'.format(i)
            acc_list = np.load(osp.join(acc_path, acc_file))
            vari_file = 'vari{}.npy'.format(i)
            vari = np.load(osp.join(vari_path, vari_file))

            expand_to_num = math.floor(args.end_point * args.expand_rate)
            query_start_num = math.floor(args.end_point * args.query_rate)
            if expand_to_num >= list_length:
                expand_to_num = list_length - 1

            index1 = np.argsort(dists)
            vari_sort_distance = vari[index1]

            x = np.arange(list_length)
            acc_sort_distance = acc_list[index1]
            FNacc1 = np.zeros(list_length)
            for j in range(list_length):  # 依次求acc
                FNacc1[j] = acc_sort_distance[0:j + 1].sum() / (j + 1)
            lns2 = ax2.plot(x[args.start_point:expand_to_num], FNacc1[args.start_point:expand_to_num],
                            label='model_{}_FNacc'.format(i))
            lns3 = ax2.plot(x[args.start_point:expand_to_num],
                            acc_sort_distance[args.start_point:expand_to_num] * 0.2,
                            'bo',
                            ms=1, color='g', label='model_{}_acc_list'.format(i))
            ax1 = ax2.twinx()
            lns1 = ax1.plot(x[args.start_point:expand_to_num], vari_sort_distance[args.start_point:expand_to_num],
                            color='coral',
                            label='model_{}_vari'.format(i))

            # resort by variance in [query_start_num:expand_to_num]
            acc1 = acc_sort_distance[:args.end_point].sum()  #
            vari_range = vari_sort_distance[:expand_to_num]  #
            index2 = np.argsort(-vari_range)  # down sort
            second_sampling_index2 = index2[:args.end_point]
            acc_sort_variance = acc_sort_distance[second_sampling_index2]
            acc2 = acc_sort_variance.sum()  # /second_samping_num
            # vari_range2 = vari_sort_distance[query_start_num:expand_to_num]
            index3 = np.argsort(-vari_sort_distance[query_start_num:expand_to_num]) + query_start_num
            second_sampling_index3 = np.append(np.arange(query_start_num),
                                               index3[:args.end_point - query_start_num])
            acc_sort_variance3 = acc_sort_distance[second_sampling_index3]
            acc3 = acc_sort_variance3.sum()
            print('second_sampling_index3\'s length ={}'.format(len(second_sampling_index3)))
            print(
                'model_{} :: acc1:{}/{:.2%}, acc2:{}/{:.2%} acc3:{}/{:.2%}'.format(i, acc1, acc1 / args.end_point,
                                                                                   acc2,
                                                                                   acc2 / args.end_point, acc3,
                                                                                   acc3 / args.end_point))
            acc_file = codecs.open(acc_file_name, mode='a')
            acc_file.write('{}/{:.2%},{}/{:.2%},{}/{:.2%}\n'.format(acc1, acc1 / args.end_point,
                                                                    acc2,
                                                                    acc2 / args.end_point, acc3,
                                                                    acc3 / args.end_point))

            vari_flag = vari_sort_distance[:expand_to_num]
            for j in range(expand_to_num):
                if j not in second_sampling_index3:
                    # print("not in index2")
                    vari_flag[j] = 0.3
            lns4 = ax1.plot(x[args.start_point:expand_to_num], vari_flag, 'bo', ms=1,
                            label='model_{}_sec_samp'.format(i),
                            color='r')
            lns = lns1 + lns2 + lns3 + lns4
            labs = [l.get_label() for l in lns]
            # ax1.legend(lns, labs, loc='best')
            if i == len(query_rate_list)-1:
                ax1.legend(lns, labs, loc='lower right', bbox_to_anchor=(2, 0))
            ax1.set_xlabel('distance')
            ax1.set_ylabel('variance')
            ax2.set_ylabel('FN_acc')
        plt.savefig(osp.join(input_path,'{}_of_EF15_{}'.format(key_name,args.function)))


    elif args.function == 12: # based on function 9 and 10
        best_parameter_file = codecs.open(osp.join(input_path,'best_parameter_11.txt'),'r','utf-8')
        q_half_e_file = codecs.open(osp.join(input_path,'q_half_e_11.txt'),'r','utf-8')
        bp_infor = best_parameter_file.readlines()
        bp_model_list = []
        for line in bp_infor:
            acc_value = []
            infor = line.strip().split(',')
            for acc_info in infor:
                value = int(acc_info.split('/')[0].strip('.0'))
                percent = float(acc_info.split('/')[1].strip('%'))
                acc_value.extend([value, percent])
            bp_model_list.append(acc_value)
        qe_infor = q_half_e_file.readlines()
        qe_model_list = []
        for line in qe_infor:
            acc_value = []
            infor = line.strip().split(',')
            for acc_info in infor:
                value = int(acc_info.split('/')[0].strip('.0'))
                percent = float(acc_info.split('/')[1].strip('%'))
                acc_value.extend([value, percent])
            qe_model_list.append(acc_value)
        # until there ,all the data hava been extracted.
        bp_model_list = np.array(bp_model_list)
        qe_model_list = np.array(qe_model_list)
        plt.figure(figsize=(20, 14), dpi=300)
        for model in range(len(bp_model_list)):
            plt.subplot(3,3,model+1)
            x = np.arange(3)
            y1 = bp_model_list[model,[1,3,5]]
            y2 = qe_model_list[model,[1,3,5]]
            plt.plot(x,y2,label = 'q_half_e')
            plt.plot(x, y1, label='best_parameter')
            plt.xlabel('model{}'.format(model+1))
            plt.legend(loc='best')
        plt.savefig(osp.join(input_path, 'analysisi_{}'.format(args.function)))

    elif args.function ==15:
        '''
               draw picture for vcfs to illustrate the shortcoming of distance confidence.
               '''
        model_order  =0
        sampling_num = 300
        dists_file = 'dists{}.npy'.format(model_order)
        dists = np.load(osp.join(dists_path, dists_file))
        list_length = dists.shape[0]
        acc_file = 'acc_list{}.npy'.format(model_order)
        acc_list = np.load(osp.join(acc_path, acc_file))
        vari_file = 'vari{}.npy'.format(model_order)
        vari = np.load(osp.join(vari_path, vari_file))
        index1 = np.argsort(dists)
        select_index  =  index1[:sampling_num]
        y_dists = dists[select_index]
        y_acc = acc_list[select_index]

        total_right_num = sum(acc_list)
        select_right_num = sum(y_acc)

        for idx in range(sampling_num):
            if y_acc[idx] == 1:
                y_acc[idx]=0
            else: y_acc[idx] = 0.1

        print("there are {} samples have right label in all, when we select {} samples, there are {} samples have rigth label.".format(total_right_num,sampling_num,select_right_num))
        x = np.arange(sampling_num)
        plt.figure(figsize=(6, 4), dpi=300)
        plt.plot(x,y_dists,label='distance')
        plt.bar(x,y_acc,label ='wrong estimation',color='r')
        plt.legend(loc='best')
        plt.xlabel('Unlabel samples')
        plt.ylabel('Distance')
        # plt.title('The insufficient of distance-based sampling criterion')
        plt.savefig('vcfs_image_{}'.format(args.function))


        expand_rate = 0.65
        expande_num = math.floor(sampling_num / expand_rate)
        first_index = index1[:expande_num]
        vari_first = vari[first_index]
        index2 = np.argsort(-vari_first)
        second_index = first_index[index2[:sampling_num]]
        y_vari = vari[second_index]
        y_acc2 = acc_list[second_index]

        select_right_num2 = sum(y_acc2)
        print(
            "there are {} samples have right label in all, when we select {} samples, there are {} samples have rigth label.".format(
                total_right_num, sampling_num, select_right_num2))

        for idx in range(sampling_num):
            if y_acc2[idx] == 1:
                y_acc2[idx]=0
            else: y_acc2[idx] = 0.05
        x = np.arange(sampling_num)
        plt.figure(figsize=(6, 4), dpi=300)
        plt.plot(x, y_vari, label='variance')
        plt.bar(x, y_acc2, label='wrong estimation', color='r')
        plt.legend(loc='best')
        plt.xlabel('Unlabel samples')
        plt.ylabel('Variance')
        # plt.title('The insufficient of distance-based sampling criterion')
        plt.savefig('vcfs_image_{}2'.format(args.function))





    #     expand_to_num = math.floor(args.end_point * args.expand_rate)
    #     query_start_num = math.floor(args.end_point * args.query_rate)
    #     if expand_to_num >= list_length:
    #         expand_to_num = list_length - 1
    #
    #     index1 = np.argsort(dists)
    #     vari_sort_distance = vari[index1]
    #
    #     x = np.arange(list_length)
    #     acc_sort_distance = acc_list[index1]
    #     FNacc1 = np.zeros(list_length)
    #     for j in range(list_length):  # 依次求acc
    #         FNacc1[j] = acc_sort_distance[0:j + 1].sum() / (j + 1)
    #     lns2 = ax2.plot(x[args.start_point:expand_to_num], FNacc1[args.start_point:expand_to_num],
    #                     label='model_{}_FNacc'.format(i))
    #     lns3 = ax2.plot(x[args.start_point:expand_to_num],
    #                     acc_sort_distance[args.start_point:expand_to_num] * 0.2,
    #                     'bo',
    #                     ms=1, color='g', label='model_{}_acc_list'.format(i))
    #     ax1 = ax2.twinx()
    #     lns1 = ax1.plot(x[args.start_point:expand_to_num], vari_sort_distance[args.start_point:expand_to_num],
    #                     color='coral',
    #                     label='model_{}_vari'.format(i))
    #
    #     # resort by variance in [query_start_num:expand_to_num]
    #     acc1 = acc_sort_distance[:args.end_point].sum()  #
    #     vari_range = vari_sort_distance[:expand_to_num]  #
    #     index2 = np.argsort(-vari_range)  # down sort
    #     second_sampling_index2 = index2[:args.end_point]
    #     acc_sort_variance = acc_sort_distance[second_sampling_index2]
    #     acc2 = acc_sort_variance.sum()  # /second_samping_num
    #     # vari_range2 = vari_sort_distance[query_start_num:expand_to_num]
    #     index3 = np.argsort(-vari_sort_distance[query_start_num:expand_to_num]) + query_start_num
    #     second_sampling_index3 = np.append(np.arange(query_start_num),
    #                                        index3[:args.end_point - query_start_num])
    #     acc_sort_variance3 = acc_sort_distance[second_sampling_index3]
    #     acc3 = acc_sort_variance3.sum()
    #     print('second_sampling_index3\'s length ={}'.format(len(second_sampling_index3)))
    #     print(
    #         'model_{} :: acc1:{}/{:.2%}, acc2:{}/{:.2%} acc3:{}/{:.2%}'.format(i, acc1, acc1 / args.end_point,
    #                                                                            acc2,
    #                                                                            acc2 / args.end_point, acc3,
    #                                                                            acc3 / args.end_point))
    #     acc_file = codecs.open(acc_file_name, mode='a')
    #     acc_file.write('{}/{:.2%},{}/{:.2%},{}/{:.2%}\n'.format(acc1, acc1 / args.end_point,
    #                                                             acc2,
    #                                                             acc2 / args.end_point, acc3,
    #                                                             acc3 / args.end_point))
    #
    #     vari_flag = vari_sort_distance[:expand_to_num]
    #     for j in range(expand_to_num):
    #         if j not in second_sampling_index3:
    #             # print("not in index2")
    #             vari_flag[j] = 0.3
    #     lns4 = ax1.plot(x[args.start_point:expand_to_num], vari_flag, 'bo', ms=1,
    #                     label='model_{}_sec_samp'.format(i),
    #                     color='r')
    #     lns = lns1 + lns2 + lns3 + lns4
    #     labs = [l.get_label() for l in lns]
    #     # ax1.legend(lns, labs, loc='best')
    #     if i == len(query_rate_list) - 1:
    #         ax1.legend(lns, labs, loc='lower right', bbox_to_anchor=(2, 0))
    #     ax1.set_xlabel('distance')
    #     ax1.set_ylabel('variance')
    #     ax2.set_ylabel('FN_acc')
    # plt.savefig(osp.join(input_path, '{}_of_EF15_{}'.format(key_name, args.function)))







if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='analysis_data_handle')
    parser.add_argument('--function',type=int,default=7)
    parser.add_argument('--model_order',type=int,default=0)
    parser.add_argument('--second_sampling_rate',type=float,default=0.8)
    parser.add_argument('--expand_rate',type=float,default=1.1)
    parser.add_argument('--end_point',type=int,default=-1)
    parser.add_argument('--start_point',type=int,default=0)
    parser.add_argument('--is_best_para',type=int,default=1) #1 indecate is best_parameter, set for function 11
    parser.add_argument('--query_rate',type=float,default=0.5)
    working_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--data_dir', type=str, metavar='PATH', default=os.path.join(working_dir, 'data'))  # 加载数据集的根目录
    parser.add_argument('--logs_dir', type=str, metavar='PATH', default=os.path.join(working_dir, 'logs'))  # 保持日志根目录
    parser.add_argument('--exp_name', type=str, default="epsm")
    parser.add_argument('--exp_order', type=str, default="0")
    parser.add_argument('--dataset', type=str, default='DukeMTMC-VideoReID')
    main(parser.parse_args())

    '''
        文件描述 :
        draw picture for vcfs. 
        运行命令如:
        # python3.6  analysis_data_handle.py --exp_name gradually_11step --exp_order 0
        # python3.6  analysis_data_handle.py --exp_name gradually_EF15 --exp_order 0  --function 11 --is_best_para 1
        # python3.6  analysis_data_handle.py --exp_name gradually_EF15 --exp_order 0  --function 12 
        python3.6 vcfs_image.py --exp_name gradually_11step --exp_order 0 --function 15
        
        
    '''









