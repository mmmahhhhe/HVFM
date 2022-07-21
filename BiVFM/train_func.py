#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/28 14:09
# @Author  : 马赫
# @Email   : 1692303843@qq.com
# @FileName: train.py


import warnings

warnings.filterwarnings("ignore")

try:
    from BiVFM.func import *
    from BiVFM.olga_func import *
    from BiVFM.model import *
    from BiVFM.feature_engineering import *
    from BiVFM.config_transfer import *
except Exception:
    from func import *
    from olga_func import *
    from model import *
    from feature_engineering import *
    from config_transfer import *

import paddle.nn.functional as F
from paddle.io import DataLoader
from paddle.optimizer import *

# ============== 神经网络部分 ==============


def predict_loader(model, loader):
    """
    输入模型和迭代器，
    如果迭代器同时存在数据和标签，则返回预测的y和真实y，否则只返回预测的y
    y二维矩阵[时间步, 特征维度]
    """
    # TODO 不知道为什么，这样next迭代每次迭代都会增加内存占用

    # # 是否同时存在数据和标签
    # for batch_id, _data in enumerate(loader()):
    #     break

    # if len(_data) == 2:

    # # 只有数据的情况
    # else:
    #     predict_list = []
    #     for batch_id, data in enumerate(loader()):
    #         data = data.astype('float32')
    #         output = model(data)
    #         predict_list.append(output.numpy())
    #     return np.vstack(predict_list)

    predict_list = []
    y_list = []
    for batch_id, (data, label) in enumerate(loader()):
        data = data.astype('float32')
        label = label.astype('float32')
        output = model(data)

        predict_list.append(output.numpy())
        y_list.append(label.numpy())
    return np.vstack(predict_list), np.vstack(y_list)


class MyIterableDataset(paddle.io.IterableDataset):
    def __init__(self, data_list, x_length, y_length, x_features, y_features, step=1, return_y_info=False):
        """
        输入标准化后的数据
        """
        self.data_list = data_list
        self.x_length = x_length
        self.y_length = y_length
        self.x_features = x_features
        self.y_features = y_features
        self.step = step
        # 是否返回输出y时刻的x信息
        self.return_y_info = return_y_info

    def __iter__(self):

        for data in self.data_list:
            # 返回输出y时刻的x信息
            if self.return_y_info:
                x_window, y_window, y_x_info = generate_xy(data=data,
                                                           x_length=self.x_length, y_length=self.y_length,
                                                           x_features=self.x_features, y_features=self.y_features,
                                                           step=self.step,
                                                           return_y_info=self.return_y_info)
                for x, y, z in zip(x_window, y_window, y_x_info):
                    yield x, y, z

            # 不返回输出y时刻的x信息
            else:
                x_window, y_window = generate_xy(data=data,
                                                 x_length=self.x_length, y_length=self.y_length,
                                                 x_features=self.x_features, y_features=self.y_features,
                                                 step=self.step,
                                                 return_y_info=self.return_y_info)
                for x, y in zip(x_window, y_window):
                    yield x, y


def load_params(model, model_path='save_dir/net.pdparams', opt=None, opt_path='save_dir/opt.pdopts'):
    """
    模型加载
    主要保存模型本身，
    优化器可能有bug建议不读取优化器opt
    """

    assert os.path.isfile(model_path), 'Failed to load model'
    layer_state_dict = paddle.load(model_path)

    model.set_state_dict(layer_state_dict)
    if opt != None:
        assert os.path.isfile(opt_path), 'Failed to load optimizer'
        opt_state_dict = paddle.load(opt_path)
        opt.set_state_dict(opt_state_dict)


def save_params(model, model_path='save_dir/net.pdparams', opt=None, opt_path='save_dir/opt.pdopts'):
    """
    模型保存
    主要保存模型本身，
    优化器可能有bug建议不保存优化器opt
    """

    paddle.save(model.state_dict(), model_path)
    if opt != None:
        paddle.save(opt.state_dict(), opt_path)


class LossFC(paddle.nn.Layer):
    def __init__(self):
        super(LossFC, self).__init__()
        pass

    def forward(self, data, label):
        loss = F.mse_loss(data, label, reduction='mean')
        return loss


def train_model(net, opt, loss_fc,
                num_epoch, continue_epoch,
                train_loader, test_loader,
                scalar_s2r_y,
                y_feature,
                base_conf, feature_list,
                is_show=True
                ):
    """
    net:模型
    opt:模型
    loss_fc:损失函数

    num_epoch: int 总轮次
    continue_epoch: int 从第几轮次继续

    train_loader:训练迭代器
    test_loader:验证绘图迭代器

    scalar_s2r_y: 元组 (全部y的均值向量，全部y的方差向量) 将标准化的y转化为原始尺度

    y_feature: list 矩阵的哪几列是y，一般为最后4列

    base_conf: dict 配置
    feature_list: np.array 全部列名向量

    is_show: 是否显示


    """

    min_loss = 1e9  # 初始化 最小误差
    for epoch in range(continue_epoch, num_epoch):
        net.train()
        loss_list = []
        for batch_id, (data, label) in enumerate(train_loader()):
            data = data.astype('float32')
            label = label.astype('float32')
            output = net(data)

            step_loss = loss_fc(output, label)

            loss_list.append(step_loss.numpy())
            step_loss.backward()
            opt.step()
            opt.clear_grad()
            if batch_id % 100 == 0:
                # 每n轮训练保存一次日志
                loss = sum(loss_list) / len(loss_list)
                log("Training epoch: %s batch_id: %s, loss: %s" % (epoch, batch_id, loss),
                    log_file=base_conf['train_log_file'])

        loss = sum(loss_list) / len(loss_list)

        if loss <= min_loss:
            # 当前损失值小于历史最低的损失值时，进行测试
            # 保存模型，并可视化，记录日志

            # 将模型转为测试模式
            net.eval()
            # 保存模型的网络结构参数
            save_params(net, model_path=base_conf['model_path'])

            # 获取训练过程的图像保存路径
            save_path = os.path.join(base_conf['save_path'], '%s.jpg' % epoch)
            # 训练过程的训练效果可视化
            visualization(model=net, loader=test_loader,
                          standardized_args=scalar_s2r_y,
                          is_show=is_show, save_path=save_path,
                          title=feature_list[y_feature],
                          show_error=False)

            # 将每次获得最佳成绩的记录保存日志
            content = "Current Epoch: %s, loss %s  better than last loss %s, save model" % (epoch, loss, min_loss)
            log(content, log_file=base_conf['train_log_file'])
            min_loss = loss

        if epoch % 1 == 0:
            # 每n轮训练保存一次日志
            log("Training epoch: %s, loss: %s" % (epoch, loss), log_file=base_conf['train_log_file'])

    log("Training till end", log_file=base_conf['train_log_file'])


def train_process(base_yaml_path):
    """
    One button to start training
    :param base_yaml_path:str, config file absolute path
    :return: None
    """

    base_conf = get_base_config(base_yaml_path)
    # print(base_conf)

    pkl_dir = base_conf['pkl_dir']
    num_epoch = base_conf['num_epoch']
    input_channels = base_conf['input_channels']
    output_channels = base_conf['output_channels']

    continue_epoch = base_conf['continue_epoch']  # 从多少轮开始训练

    ratio = base_conf['ratio']
    model_name = base_conf['model_name']

    load_model = base_conf['load_model']

    # 加载数据
    raw_data_list = pickle_load('%s/raw_data_list.pkl' % pkl_dir)
    print('Number of sample files:', len(raw_data_list))

    # 以全部文件作为标准化基准
    all_mean = []
    all_std = []
    for i in raw_data_list:
        mean = i.mean()
        std = i.std()
        all_mean.append(mean)
        all_std.append(std)

    all_mean = pd.concat(all_mean, axis=1).T
    all_std = pd.concat(all_std, axis=1).T

    standardized_args = (all_mean.mean(), all_std.mean())

    stand_data_list = [raw2stand(raw_data, standardized_args) for raw_data in raw_data_list]

    # 切分特征
    feature_list = np.array(raw_data_list[0].columns)

    # print('Dataset Characteristics:')
    # for num, feature in enumerate(feature_list):
    #     print(num, feature)
    # print('=' * 10)

    x_feature = list(range(len(feature_list)))[:-output_channels]
    y_feature = list(range(len(feature_list)))[-output_channels:]

    print('x特征%s列，y特征%s列' % (len(x_feature), len(y_feature)))
    print('========== x_feature: ==========\n%s' % feature_list[x_feature])
    print('========== y_feature: ==========\n%s' % feature_list[y_feature])

    np.random.seed(0)
    np.random.shuffle(stand_data_list)

    num_ratio = int(len(stand_data_list) * ratio)
    train_data_list = stand_data_list[: num_ratio]
    test_data_list = stand_data_list[num_ratio:]

    # DataLoader
    train_reader = MyIterableDataset(train_data_list, base_conf['x_length'], base_conf['y_length'], x_feature,
                                     y_feature)

    test_reader = MyIterableDataset(test_data_list[: 10], base_conf['x_length'], base_conf['y_length'],
                                    x_feature,
                                    y_feature, return_y_info=False)

    train_loader = DataLoader(train_reader, batch_size=base_conf['batch_size'], drop_last=False)
    test_loader = DataLoader(test_reader, batch_size=base_conf['batch_size'], drop_last=False)

    for batch_id, (data, label) in enumerate(train_loader()):
        print('train batch x, y shape\t', data.shape, label.shape)
        break

    # scalar_s2r = None
    # scalar_s2r_x = (standardized_args[0].iloc[x_feature].values, standardized_args[1].iloc[x_feature].values)
    scalar_s2r_y = (standardized_args[0].iloc[y_feature].values, standardized_args[1].iloc[y_feature].values)

    net = Net(input_channels, output_channels, time_steps=base_conf['y_length'], model=model_name)

    opt = Adam(learning_rate=base_conf['learning_rate'], parameters=net.parameters())  # Adam

    loss_fc = LossFC()

    # paddle.summary(Net(input_channels, output_channels), (128, 187, 5))

    # a = Net(input_channels, time_steps=1)
    # shape = (128, 187, 35)
    # print('input shape', shape)
    # print('output shape', a(paddle.randn(shape)).shape)
    # paddle.summary(a, shape)

    if load_model:
        try:
            load_params(net, model_path=base_conf['model_path'])
            log('模型已加载', log_file=base_conf['train_log_file'])
        except Exception as e:
            log('未加载模型', log_file=base_conf['train_log_file'])

    else:
        log('未加载模型', log_file=base_conf['train_log_file'])

    train_model(net=net, opt=opt, loss_fc=loss_fc, num_epoch=num_epoch, continue_epoch=continue_epoch,
                train_loader=train_loader, test_loader=test_loader, scalar_s2r_y=scalar_s2r_y, y_feature=y_feature,
                base_conf=base_conf, feature_list=feature_list, is_show=True)
