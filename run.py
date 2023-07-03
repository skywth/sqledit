from src.preproc.preproc import preproc
from src.train.train_codet5 import train_codet5
from src.eval.eval_codet5 import eval_codet5

import argparse        #用于命令项选项与参数解析的模块
import numpy as np
import random
import torch


def add_general_args(parser):
    parser.add_argument(
        "--program_only", action="store_true"
    )
    parser.add_argument(
        "--seed", type=int, 
        default=42
    )
    parser.add_argument(
        "--gpu", type=int, 
        default=0
    )
    parser.add_argument(
        "--beam_size", type=int,  #束搜索的束宽
        default=3
    )
    return parser


def add_file_args(parser):
    parser.add_argument(
        "--spider_train_fname", type=str, 
        default="data/spider/train_spider.json"   #spider的训练集
    )
    parser.add_argument(
        "--spider_dev_fname", type=str, 
        default="data/spider/dev.json"            #spider的验证集
    )
    parser.add_argument(
        "--sqledit_train_fname", type=str, 
        default="data/sqledit-train.json"          #修改训练使用数据集
    )
    parser.add_argument(
        "--sqledit_dev_fname", type=str, 
        default="data/sqledit-dev.json"            #修改验证使用数据集
    )
    parser.add_argument(
        "--sqledit_test_fname", type=str, 
        default="data/sqledit-test.json"            #测试集
    )
    parser.add_argument(
        "--load_checkpoint", type=str, 
        default="Salesforce/codet5-base"            #使用模型coder-base模型
    )
    parser.add_argument(
        "--save_checkpoint", type=str, 
        default="model/"                            #模型保存
    )
    return parser


def add_preproc_args(parser):
    parser.add_argument(
        "--preproc", action="store_true"            #加载数据
    )
    parser.add_argument(
        "--use_content", action="store_true"
    )
    parser.add_argument(
        "--query_type", type=str, 
        default="pydict"                       #问题类型
    )
    parser.add_argument(
        "--edit_type", type=str, 
        default="program"                       #修改类型
    )
    parser.add_argument(
        "--base_parser", type=str, 
        default="codet5"                        #模型
    )
    return parser


def add_train_args(parser):
    parser.add_argument(
        "--train", action="store_true"        #训练类型
    )
    parser.add_argument(
        "--epochs", type=int,               #轮次
        default=10
    )
    parser.add_argument(
        "--lr", type=float,                   #学习率
        default=3e-5
    )
    parser.add_argument(
        "--batch_size", type=int,         #单次传递给程序用以训练的数据（样本）个数
        default=4
    )
    parser.add_argument(
        "--grad_accum", type=int,        #每次处理完一个batch，不直接更新梯度。而是计算多个batch，将梯度进行累加，再做梯度更新
        default=2                        #处理2个batch进行梯度更新
    )
    return parser


def add_eval_args(parser):
    parser.add_argument(
        "--eval", action="store_true"                #评估
    )
    parser.add_argument(
        "--etype", type=str,
        default="all"                     #all, exec, match代表EX/EM or EX&EM
    )
    return parser


def get_args():
    parser = argparse.ArgumentParser()
    parser = add_general_args(parser)
    parser = add_file_args(parser)
    parser = add_preproc_args(parser)
    parser = add_train_args(parser)
    parser = add_eval_args(parser)
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)

    if args.preproc:        #选择要进行操作的类型
        preproc(args)
    elif args.train:
        train_codet5(args)
    elif args.eval:
        eval_codet5(args)
    else:
        print("Error: must specify preproc/train/eval in args")
