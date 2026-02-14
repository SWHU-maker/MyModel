import argparse # 导入参数解析模块，用来处理你在命令行输入的各种参数（如 --batch_size）
import torch # 导入 PyTorch 框架
from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast # 导入长程预测实验类
import random # 导入随机数模块
import numpy as np # 导入数值计算模块

if __name__ == '__main__':
    # 为了保证实验结果可以“复现”（即每次跑出的准确率都一样），需要固定随机种子
    # fix_seed = 2023
    # random.seed(fix_seed) # 设置 Python 自带的随机种子
    # torch.manual_seed(fix_seed) # 设置 PyTorch 的随机种子
    # np.random.seed(fix_seed) # 设置 Numpy 的随机种子

    # 创建一个解析器对象，description 是对本项目的简单描述
    parser = argparse.ArgumentParser(description='PatchMLP')
    parser.add_argument('--random_seed', type=int, default=2023, help='random seed')

    # 【基础配置参数】
    # --is_training: 是训练还是测试？1 表示训练，0 表示直接跑测试集
    parser.add_argument('--is_training', type=int,  default=1, help='status')
    # --model_id: 给你这次实验取个名字，方便后面在文件夹里找结果
    parser.add_argument('--model_id', type=str,  default='test', help='model id')
    # --model: 使用哪个模型？默认是 PatchMLP
    parser.add_argument('--model', type=str,  default='PatchMLP', help='model name')

    # 【模型超参数配置】
    # --dropout: 随机丢弃一部分神经元防止过拟合的比例
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    # --d_model: 模型内部向量的维度，越大模型越宽，但也越慢
    parser.add_argument('--d_model', type=int, default=1024, help='dimension of model')
    # --moving_avg: 移动平均的窗口大小，用于平滑序列趋势
    parser.add_argument('--moving_avg', type=int, default=13, help='window size of moving average')
    # --e_layers: 编码器（Encoder）的层数
    parser.add_argument('--e_layers', type=int, default=1, help='num of encoder layers')
    # --use_norm: 是否使用归一化和逆归一化（RevIN），对处理非平稳数据很有帮助
    parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm')
    # --train_epochs: 训练多少轮
    parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
    # --batch_size: 每次训练时同时处理多少条数据
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    # --patience: 如果验证集 Loss 连续多少轮不下降，就提前停止（早停机制）
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    # --learning_rate: 学习率，决定了模型参数更新的步长
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')

    # 【数据加载相关配置】
    # --data: 数据集类型（如 custom, ETTh1 等）
    parser.add_argument('--data', type=str, default='custom', help='dataset type')
    # --root_path: 数据集文件所在的文件夹路径
    parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
    # --data_path: CSV 数据文件的具体名称
    parser.add_argument('--data_path', type=str, default='weather.csv', help='data csv file')
    # --features: 预测任务类型。M:多变多, S:单变单, MS:多变单
    parser.add_argument('--features', type=str, default='M', help='forecasting task')
    # --target: 在 S 或 MS 任务中，你想预测哪一列数据
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    # --freq: 时间频率（如 h:小时, t:分钟, d:天）
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding')
    # --checkpoints: 训练好的模型权重存在哪里
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # 【预测任务长度配置】
    # --seq_len: 输入给模型的历史序列长度（看过去多远）
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    # --label_len: 引导序列的长度（Decoder 输入的前半部分）
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    # --pred_len: 预测未来的长度（预测多远）
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # 【模型结构详细定义】
    # --enc_in: 编码器的输入维度（有多少列特征）
    parser.add_argument('--enc_in', type=int, default=21, help='encoder input size')
    # --dec_in: 解码器的输入维度
    parser.add_argument('--dec_in', type=int, default=21, help='decoder input size')
    # --c_out: 输出的维度
    parser.add_argument('--c_out', type=int, default=21, help='output size')
    # --embed: 时间特征的编码方式（默认 timeF）
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')
    # --do_predict: 训练完后是否进行真实的未来预测
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # 【优化与硬件配置】
    # --num_workers: 数据加载的线程数，越多加载越快，但吃内存
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    # --itr: 实验重复跑几次（为了求平均值减小偶然性）
    parser.add_argument('--itr', type=int, default=26, help='experiments times')
    # --use_gpu: 是否使用显卡计算
    parser.add_argument('--use_gpu', type=bool, default=False, help='use gpu')
    # --gpu: 指定使用哪块显卡（0 号、1 号等）
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    # --use_multi_gpu: 是否使用多卡并行
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    # --devices: 多卡并行的显卡编号列表
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # 解析所有输入的参数，存入 args 变量
    args = parser.parse_args()
    # 检查硬件：如果显卡可用且用户想用，就设为 True，否则强行设为 False
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    # 处理多显卡的编号字符串
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args) # 在控制台打印出这次运行的所有参数配置

    # 设置实验类为长程预测
    Exp = Exp_Long_Term_Forecast

    if args.is_training: # 如果是训练模式
        for ii in range(args.itr):
            # 自动生成一个超级长的文件夹名字，包含所有核心参数，防止实验结果混淆
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
                args.model_id, args.model, args.data, args.features,
                args.seq_len, args.label_len, args.pred_len,
                args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff,
                args.factor, args.embed, args.distil, args.des, ii)

            exp = Exp(args)  # 实例化实验（也就是 self 诞生的时刻）
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting) # 开始训练流程

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting) # 训练完后跑测试集

            if args.do_predict: # 如果需要预测未来
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache() # 跑完一次实验清理一下显存，防止溢出
    else: # 如果是测试模式（直接加载已经训练好的模型）
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.model_id, args.model, args.data, args.features,
            args.seq_len, args.label_len, args.pred_len,
            args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff,
            args.factor, args.embed, args.distil, args.des, args.class_strategy, ii)

        exp = Exp(args)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1) # 执行测试逻辑
        torch.cuda.empty_cache()