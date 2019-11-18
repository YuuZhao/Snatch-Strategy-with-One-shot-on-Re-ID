import argparse


def main(args):


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Snatch Strategy')
    parser.add_argument('--dataset', type=str,default='DukeMTMC-VideoReID')  # 加载数据集的根目录
    parser.add_argument('--exp_name',type=str,default="exp_test")
    parser.add_argument('--function', type=int, default=None)


    #下面是暂时不知道用来做什么的参
    main(parser.parse_args())
