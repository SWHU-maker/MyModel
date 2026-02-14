# 导入本项目自定义的数据模块。data_provider 是一个“工厂函数”，它会根据你输入的参数（比如是哪个数据集）
# 自动帮你准备好 PyTorch 训练需要的 DataLoader（数据加载器）
from data_provider.data_factory import data_provider

# 导入基础实验类 Exp_Basic。它定义了实验的通用流程（比如选 CPU 还是 GPU），
# 这里的 Exp_Long_Term_Forecast 继承它，可以少写很多重复代码
from experiments.exp_basic import Exp_Basic

# 导入工具函数：
# EarlyStopping: “早停”机制。如果模型在验证集上不再进步，就提前停止训练，防止过拟合。
# adjust_learning_rate: 动态调整学习率。训练后期让学习率变小，使模型收敛更稳定。
# visual: 画图工具，把预测结果和真实值画在一起看直观效果。
from utils.tools import EarlyStopping, adjust_learning_rate, visual

# 导入评价指标计算函数（计算 MSE, MAE 等），用来评估预测得准不准
from utils.metrics import metric

import torch
import torch.nn as nn
from torch import optim # 包含各种优化算法，比如 Adam
import os
import time
import warnings
import numpy as np

# 忽略代码运行中的一些琐碎警告，让控制台输出更整洁
warnings.filterwarnings('ignore')

# 定义长程预测实验类。它是整个训练和测试流程的“指挥官”
class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        # super() 是调用父类 Exp_Basic 的初始化方法，确保设备（GPU/CPU）和模型基础配置已就绪
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        # 这一行是真正创建模型的地方。self.model_dict 存了模型名到类的映射。
        # .Model(self.args) 把你在 run.py 配置的所有超参数传给模型，.float() 确保使用单精度浮点数
        model = self.model_dict[self.args.model].Model(self.args).float()

        # 如果你有多个 GPU，这一行会让模型在多个 GPU 上同时跑，提高速度
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        # flag 参数通常是 'train'（训练）、'val'（验证）或 'test'（测试）。
        # data_provider 会根据 flag 返回对应的 Dataset（数据实体）和 DataLoader（批量加载器）
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # 选择 Adam 优化器。它是目前最主流的优化器，负责根据误差（Loss）来更新模型的参数
        # lr=self.args.learning_rate 是设置初始学习率（步长）
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # 设置“损失函数”。MSELoss（均方误差）是时间序列预测最常用的，
        # 它会计算预测值和真实值之间差的平方。差越大，Loss 越高
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        # 验证函数：在每个 epoch（轮次）训练完后跑一次，看看模型在没见过的数据上表现如何
        total_loss = []
        self.model.eval() # 【重要】将模型设为“评估模式”，这会关闭 Dropout 等只在训练时用的层
        
        with torch.no_grad(): # 验证阶段不需要更新参数，所以关闭梯度计算，节省内存和速度
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                # batch_x: 历史数据, batch_y: 待预测的未来真值（包含引导段）
                # batch_x_mark/y_mark: 对应的时间戳特征（如几点、周几）
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                # 特殊处理：Solar 数据集通常没有时间戳信息
                if 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # 【难点：Decoder输入构造】
                # 1. 先建一个全 0 的张量，大小和我们要预测的部分（pred_len）一样
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                # 2. 把引导段（label_len）和这个全 0 预测段拼在一起。
                # 这样模型能看到一段已知的历史，再往后预测已变零的部分
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # 模型跑一遍（前向传播）
                if self.args.use_amp: # AMP 是自动混合精度，能让高端显卡跑得更快
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                # 我们只关心预测出的那部分（最后 pred_len 个点）
                f_dim = -1 if self.args.features == 'MS' else 0 # MS 模式只取最后一个维度（目标列）
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                loss = criterion(outputs.detach().cpu(), batch_y.detach().cpu()) # 计算这一个 batch 的误差
                total_loss.append(loss)
        
        total_loss = np.average(total_loss) # 把所有 batch 的误差平均一下
        self.model.train() # 【重要】验证完一定要切回“训练模式”
        return total_loss

    def train(self, setting):
        # 准备数据加载器
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        # 创建保存模型权重的文件夹
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time() # 记录当前时间，用来算进度
        train_steps = len(train_loader) # 一个轮次要跑多少步
        # 初始化早停类，如果连续 patience 次验证 Loss 没降，就停掉
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer() # 选优化器
        criterion = self._select_criterion() # 选损失函数

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler() # 配合混合精度使用的缩放工具

        # 开始大循环，一次 epoch 就是把整个训练集数据完整看一遍
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train() # 确保在训练模式
            epoch_time = time.time()
            
            # 内部小循环，一次 batch 就是处理一小块数据（比如 32 条）
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad() # 【必做】每一小步开始前，把之前的梯度清零，否则会累加
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # 构造解码器输入（同 vali 函数）
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # 前向传播：数据进模型，得到预测值 outputs
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y) # 计算这步的损失
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                # 每隔 100 步打印一次此时的损失和运行速度，方便观察
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # 反向传播：根据损失计算每个参数该怎么调
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim) # 更新参数
                    scaler.update()
                else:
                    loss.backward() # 计算梯度
                    model_optim.step() # 更新参数

            # 跑完一个 epoch，打印耗时并去跑验证集和测试集
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            
            # 判断要不要早停。如果当前验证集效果最好，它会自动帮我们保存一个 checkpoint.pth
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # 每一轮结束，按策略调低学习率（通常是变小）
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        # 全部轮次跑完，把刚才保存的表现最好的那个模型重新加载回来，准备最后的测试
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        # 这个函数是在训练完成后，用全新的测试集数据来给模型“打分”
        test_data, test_loader = self._get_data(flag='test')
        if test: # 如果你只是想直接测已有模型，不训练，就走这一步加载模型
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/' # 图片保存路径
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                # ...（这里的逻辑和 vali 一样，就是把数据喂给模型得到结果）...
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                outputs = outputs.detach().cpu().numpy() # 转成 numpy 方便处理
                batch_y = batch_y.detach().cpu().numpy()
                
                # 如果你在训练前对数据做了归一化，现在得把它还原回去，否则画出来的图数值不对
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                preds.append(outputs)
                trues.append(batch_y)
                
                # 每隔 20 步画一张图，看看预测值（pd）和真实值（gt）贴合得紧不紧
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    # 拼接历史和未来，画出完整的曲线
                    gt = np.concatenate((input[0, :, -1], batch_y[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], outputs[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        # 整理所有的预测结果
        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        # 保存各种 numpy 数组文件，方便你以后用别的脚本做分析
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 调用 metric 计算最终的平均误差（MSE, MAE等）
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        
        # 把这些得分记录在一个文本文件里
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae) + '\n\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        return