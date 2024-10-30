import argparse
import torch
from utils.str2bool import str2bool
from exp.exp_main import Exp_Main
import random
import numpy as np
import os
import time


fix_seed = 2023
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='DeformableTST')

parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='DeformableTST', help='model name, options: [DeformableTST]')
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--features', type=str, default='M',help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--embed', type=str, default='timeF',help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--seq_len', type=int, default = 768, help='input length')
parser.add_argument('--label_len', type=int, default = 0, help='set as 0')
parser.add_argument('--pred_len', type=int, default = 96, help='prediction length')
parser.add_argument('--n_vars', type=int, default=7, help='number of variables in the input series')
parser.add_argument('--revin', type=int, default=1, help='use RevIN; True 1 False 0')
parser.add_argument('--revin_affine', type=int, default=0, help='use RevIN-affine; True 1 False 0')
parser.add_argument('--revin_subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract the last value')
parser.add_argument('--stem_ratio', type=int, default = 8, help='down sampling ratio in stem layer')
parser.add_argument('--down_ratio', type=int, default = 2, help='down sampling ratio in DownSampling layer between two stages')
parser.add_argument('--fmap_size', type=int, default = 768, help='feature series length')
parser.add_argument('--dims', nargs='+', type=int, default=[64, 128, 256, 512], help='dims for each stage')
parser.add_argument('--depths', nargs='+', type=int, default=[1, 1, 3, 1], help='number of Transformer blocks for each stage')
parser.add_argument('--drop_path_rate', type=float, default = 0.3, help='drop path rate')
parser.add_argument('--layer_scale_value', nargs='+', type=float, default=[-1, -1, -1, -1], help='layer_scale_init_value')
parser.add_argument('--use_pe', nargs='+', type=int, default=[1,1,1,1], help='use pe; True 1 False 0')
parser.add_argument('--use_lpu', nargs='+', type=int, default=[1,1,1,1], help='use Local Perception Unit; True 1 False 0')
parser.add_argument('--local_kernel_size', nargs='+', type=int, default=[3, 3, 3, 3], help='kernel size for LPU')
parser.add_argument('--expansion', type=int, default=4, help='ffn ratio')
parser.add_argument('--drop', type=float, default = 0.0, help='dropout prob for FFN module')
parser.add_argument('--use_dwc_mlp', nargs='+', type=int, default=[1,1,1,1], help='use FFN with a DWConv; True 1 False 0')
parser.add_argument('--heads', nargs='+', type=int, default=[4, 8, 16, 32], help='number of heads')
parser.add_argument('--attn_drop', type=float, default = 0.0, help='dropout prob for attention map in attention module')
parser.add_argument('--proj_drop', type=float, default = 0.0, help='dropout prob for proj in attention module')
parser.add_argument('--stage_spec', nargs='+', type=list, default=[['D'], ['D'], ['D','D','D'], ['D']], help='type of blocks in each stage')
parser.add_argument('--window_size', nargs='+', type=int, default=[3, 3, 3, 3], help='kernel size for window attention')
parser.add_argument('--nat_ksize', nargs='+', type=int, default=[3, 3, 3, 3], help='kernel size for neighborhood attention')
parser.add_argument('--ksize', nargs='+', type=int, default=[9, 7, 5, 3], help='kernel size for offset sub-network')
parser.add_argument('--stride', nargs='+', type=int, default=[8, 4, 2, 1], help='stride for offset sub-network')
parser.add_argument('--n_groups', nargs='+', type=int, default=[2, 4, 8, 16], help='number of offset groups')
parser.add_argument('--offset_range_factor', nargs='+', type=float, default=[-1, -1, -1, -1], help='restrict the offset value in a small range')
parser.add_argument('--no_off', nargs='+', type=int, default=[0,0,0,0], help='not use offset; True 1 False 0')
parser.add_argument('--dwc_pe', nargs='+', type=int, default=[0,0,0,0], help='use DWC-pe; True 1 False 0')
parser.add_argument('--fixed_pe', nargs='+', type=int, default=[0,0,0,0], help='use fixed pe; True 1 False 0')
parser.add_argument('--log_cpb', nargs='+', type=int, default=[0,0,0,0], help='use pe of SWin-v2; True 1 False 0')
parser.add_argument('--head_dropout', type=float, default = 0.1, help='dropout prob for the head')
parser.add_argument('--head_type', type=str, default='Flatten', help='Flatten')
parser.add_argument('--use_head_norm', type=int, default=1, help='use final LN layer; True 1 False 0')
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0', help='device ids of multi gpus')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')
parser.add_argument('--warmup_epochs', type=int, default=5, help='warmup epochs')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--optimizer', type=str, default='AdamW', help='type of optimizer, choose from [AdamW, Adam]')
parser.add_argument('--weight_decay', type=float, default=0.05, help='weight_decay')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--num_workers', type=int, default=10, help='number worker')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')


args = parser.parse_args()


args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
else:
    args.device_ids = [int(args.gpu)]

if __name__ == '__main__':
    print('Args in experiment:')
    print(args)

    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            setting = '{}_{}_{}_Input_{}_Output_{}_Stem_{}_Dims_{}_FFN_{}_Layer{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.seq_len,
                args.pred_len,
                args.stem_ratio,
                args.dims,
                args.expansion,
                args.depths,
                args.stage_spec,
                )

            exp = Exp(args)
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_Input_{}_Output_{}_Stem_{}_Dims_{}_FFN_{}_Layer{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.seq_len,
            args.pred_len,
            args.stem_ratio,
            args.dims,
            args.expansion,
            args.depths,
            args.stage_spec,
        )

        exp = Exp(args)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()


