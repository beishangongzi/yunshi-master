import yaml
import argparse
import os, sys
from Trainer import Trainer


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        print(f'mkdir {path}')


def get_args():
    parser = argparse.ArgumentParser(description="Runner")
    parser.add_argument('path_in', help='input path')
    parser.add_argument('path_out', help='output path')
    parser.add_argument('--local-mode', type=bool, default=True)
    # parser.add_argument('--script', type=str, default='script/eval_deeplab.yml')
    # parser.add_argument('--script', type=str, default='script/train_deeplab.yml')
    # parser.add_argument('--script', type=str, default='script/train_deeplab_34.yml')
    # parser.add_argument('--script', type=str, default='script/train_deeplab_101.yml')
    # parser.add_argument('--script', type=str, default='script/eval_deeplab_101.yml')
    # parser.add_argument('--script', type=str, default='script/train_deeplab_u.yml') # loss降不下去
    parser.add_argument('--script', type=str, default='script/train_unetpp.yml')
    # parser.add_argument('--script', type=str, default='script/eval_unetpp.yml')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    # 远程的执行
    if not args.local_mode:
        #  检查 input dir
        if not os.path.exists(args.path_in):
            print(f'{args.path_in} hasn\'t been found')
            exit(-1)
        #  检查 output dir
        mkdir(args.path_out)
        # 开始预测
        with open(args.script, encoding='utf-8') as fp:
            content = fp.read()
            config = yaml.load(content, Loader=yaml.FullLoader)
            config['dataset']['root'] = args.path_in
            config['dataset']['root_out'] = args.path_out
            trainer = Trainer(config)
            if 'train' in config['run']:
                trainer.train()
            if 'val' in config['run']:
                trainer.validate()
            if 'test' in config['run']:
                trainer.test()
            if 'predict' in config['run']:
                trainer.predict()
    else:
        # 读取配置
        with open(args.script, encoding='utf-8') as fp:
            content = fp.read()
            config = yaml.load(content, Loader=yaml.FullLoader)
            print(config)
            # TODO predict才使用
            config['dataset']['root_out'] = args.path_out
            trainer = Trainer(config)
            if 'train' in config['run']:
                trainer.train()
            if 'val' in config['run']:
                trainer.validate()
            if 'test' in config['run']:
                trainer.test()
            if 'predict' in config['run']:
                trainer.predict()
