from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                if 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs, aux_loss = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        else:
                            outputs, aux_loss = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs, aux_loss = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        outputs, aux_loss = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss.item())

                
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        # path = os.path.join(self.args.checkpoints, setting)
        # --- 新增：获取当前日期作为一级目录 ---
        date_folder = time.strftime("%m%d", time.localtime())
        path = os.path.join(self.args.checkpoints, date_folder, setting)

        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                if 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs, aux_loss = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        else:
                            outputs, aux_loss = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y) + self.args.moe_loss_weight * aux_loss
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs, aux_loss = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        outputs, aux_loss = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y) + self.args.moe_loss_weight * aux_loss
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            # test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

            # get_cka(self.args, setting, self.model, train_loader, self.device, epoch)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        # --- 新增：获取当前日期 ---
        date_folder = time.strftime("%m%d", time.localtime())

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' , date_folder, setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = os.path.join('./results/', date_folder, setting) + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                
                
                
                # # --- 捕获最后一个 batch 的前 5 个特征并展示完整升维长度 ---
                # if i == len(test_loader) - 1:
                #     # 确定要可视化的特征数（最多 5 个）
                #     num_features = min(5, batch_x.shape[-1])
                    
                #     # 数据预处理（与模型内部 forecast 逻辑对齐）
                #     x_enc = batch_x.float().to(self.device)
                #     if self.model.use_norm:
                #         means = x_enc.mean(1, keepdim=True).detach()
                #         x_enc_temp = x_enc - means
                #         stdev = torch.sqrt(torch.var(x_enc_temp, dim=1, keepdim=True, unbiased=False) + 1e-5)
                #         x_enc_temp /= stdev
                #     else:
                #         x_enc_temp = x_enc
                    
                #     # 获取升维（Embedding）后的数据 [B, D, d_model]
                #     x_emb_in = x_enc_temp.permute(0, 2, 1) # [B, D, L]
                #     embedded_data = self.model.emb(x_emb_in) # [B, D, d_model]

                #     # 循环处理每个特征列
                #     for j in range(num_features):
                #         # 1. 原始序列数据 [L]
                #         raw_feat = batch_x[0, :, j].detach().cpu().numpy() 
                        
                #         # 2. 全量升维数据 [d_model] 
                #         exp_feat_full = embedded_data[0, j, :].detach().cpu().numpy()

                #         # 绘图 A：原始序列 (长度 L)
                #         plt.figure(figsize=(10, 4))
                #         plt.plot(raw_feat, color='blue', label=f'Original Feature {j}')
                #         plt.title(f"Feature {j}: Original (Length: {len(raw_feat)})")
                #         plt.legend()
                #         plt.savefig(os.path.join(folder_path, f"visual_feat{j}_original.png"))
                #         plt.close()

                #         # 绘图 B：完整升维后的序列 (长度 d_model)
                #         plt.figure(figsize=(15, 4)) # 增加宽度以便看清较长的序列
                #         plt.plot(exp_feat_full, color='red', label=f'Embedded Full (d_model)')
                #         plt.title(f"Feature {j}: Full Expanded Latent (Length: {len(exp_feat_full)})")
                #         plt.legend()
                #         plt.savefig(os.path.join(folder_path, f"visual_feat{j}_expanded_full.png"))
                #         plt.close()

                #     print(f"Full length visualizations for {num_features} features saved in {folder_path}")
                # # ------------------------------------------------------------------------------------------------------
                
                
                
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs, aux_loss = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        else:
                            outputs, aux_loss = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs, aux_loss = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    else:
                        outputs, aux_loss = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = os.path.join('./results/', date_folder, setting) + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        

        

        # 【重点：新增 TXT 日志保存】
        
        # A. 在根目录生成总表（所有实验结果都在这一个文件里追加）
        with open("result_all_experiments.txt", 'a') as f_all:
            f_all.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + " \n")
            f_all.write(f"Setting: {setting}\n")
            f_all.write(f"Parameters: Dropout: {self.args.dropout}, BatchSize: {self.args.batch_size}, MoE_Loss_Weight: {self.args.moe_loss_weight}\n")
            f_all.write(f"MSE: {mse:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}\n")
            f_all.write("-" * 50 + "\n\n")

        # B. 在每个实验文件夹内生成详细报告
        with open(os.path.join(folder_path, "report.txt"), 'w') as f_detail:
            f_detail.write(f"Detailed Report - {setting}\n")
            f_detail.write("=" * 50 + "\n")
            # 写入所有超参数，方便回溯
            f_detail.write(f"Arguments: {str(self.args)}\n")
            f_detail.write(f"Dropout: {self.args.dropout}\n")     # 新增
            f_detail.write(f"Batch Size: {self.args.batch_size}\n") # 新增
            f_detail.write(f"MoE Loss Weight: {self.args.moe_loss_weight}\n") # 新增
            f_detail.write("-" * 50 + "\n")
            f_detail.write(f"MSE:  {mse}\n")
            f_detail.write(f"MAE:  {mae}\n")
            f_detail.write(f"RMSE: {rmse}\n")
            f_detail.write(f"MAPE: {mape}\n")
            f_detail.write(f"MSPE: {mspe}\n")

        return


    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                preds.append(outputs)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return