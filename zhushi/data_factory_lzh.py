# 从同一个文件夹下的 data_loader.py 导入各种数据集处理类
from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Solar, \
    Dataset_Pred
from torch.utils.data import DataLoader # PyTorch 官方提供的批量加载工具

# 建立一个字典，把你在命令行输入的名称（如 ETTh1）对应到具体的类（如 Dataset_ETT_hour）
data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Solar': Dataset_Solar,
    'custom': Dataset_Custom,
}

def data_provider(args, flag):
    # 根据参数选择对应的数据类
    Data = data_dict[args.data]
    # timeenc: 时间编码方式。如果 embed 参数不是 'timeF'，则设为 0（整数编码），否则为 1（特征编码）
    timeenc = 0 if args.embed != 'timeF' else 1

    # 根据 flag（是训练、测试还是预测）来设置是否打乱数据和 Batch 大小
    if flag == 'test':
        shuffle_flag = False # 测试时不需要打乱
        drop_last = True # 如果最后剩下的数据不够一个 Batch，直接丢弃
        batch_size = 1  # 测试时通常一个一个看
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred # 预测模式下强制使用专门的预测类
    else: # 训练（train）或验证（val）模式
        shuffle_flag = True # 训练必须打乱数据，防止模型死记硬背顺序
        drop_last = True
        batch_size = args.batch_size # 使用你在 run.py 设置的 batch_size
        freq = args.freq

    # 实例化数据集类：这一步会触发 data_loader.py 里的 __read_data__，把磁盘上的 CSV 加载到内存
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len], # [输入长度, 引导长度, 预测长度]
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
    )
    print(flag, len(data_set)) # 打印一下当前数据集有多少个样本
    
    # 使用 DataLoader 封装数据集，实现多线程（num_workers）批量读取
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader