#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/28 11:30
# @Author  : 马赫
# @Email   : 1692303843@qq.com
# @FileName: func.py


import os
import shutil
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

plt_config = {
    "font.family": 'Times New Roman',  # 设置字体类型
    "font.size": 13,
}
rcParams.update(plt_config)

import yaml
import pickle
import logging


# ============== 公共方法 ==============


def rm_dir(path):
    """
    保证该文件夹不存在
    """
    if os.path.isdir(path):
        shutil.rmtree(path)


def remove_list(base='olga_work', removes=['log_imgs', 'save_dir', 'imgs_save', 'logs']):
    """
    删除列表中的文件或文件夹
    """
    for i in removes:
        try:
            abs_i = '%s/%s' % (base, i)
            shutil.rmtree(abs_i)
        except Exception as e:
            pass


def log(content, log_file='logs/log.txt'):
    """
    logging库，将content内容同时输出到控制台，并追加到log_file
    """

    # 确保存在该路径
    path = os.path.split(log_file)[0]
    if not os.path.isdir(path):
        os.makedirs(path)

    logger = logging.getLogger()
    # 将logging模块的handlers列表清空
    # 这行代码很重要！！
    logger.handlers = []

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("[%(asctime)s] %(filename)s [line:%(lineno)d] %(levelname)s: %(message)s")

    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)

    logger.addHandler(sh)
    logger.addHandler(fh)

    logger.info(content)


def makedirs(path):
    """
    保证path这个文件或文件夹路径存在，如果没有则创建该路径
    path 可以为文件或文件夹
    """
    # 通过是否有.xxx后缀判断是文件或文件夹
    # 如果是文件则去除文件名
    # 如果是文件夹则直接判断该文件夹是否存在并创建
    if os.path.splitext(path)[1] != '':
        # 这个是文件
        path = os.path.split(path)[0]

    if not os.path.isdir(path):
        os.makedirs(path)


def pickle_save(path, content):
    """
    保存py变量
    """
    # 如果不存在该路径，则创建文件夹
    makedirs(path)
    with open(path, 'wb') as f:
        pickle.dump(content, f)


def pickle_load(path):
    """
    读取py变量
    """
    with open(path, 'rb') as f:
        content = pickle.load(f)
    return content


def read_df_second_resample(filename, resample_time='s'):
    """
    从csv文件读取数据，并以每秒1条记录进行插值
    第一列为时间索引，间隔可能大于或小于1秒，将其重采样到1秒
    """
    _raw_data = pd.read_csv(filename, header=0,
                            index_col=0,  # 第一列读取为时间索引
                            parse_dates=[0],
                            date_parser=df_parser
                            )

    # 原始数据时间不均匀，重采样到每秒1个点
    # raw_data = _raw_data.resample('S').mean()
    raw_data = _raw_data.resample(resample_time).mean()
    # 重采样后的缺失值进行线性插值
    raw_data = raw_data.interpolate(method='linear')
    print('原始数据形状：%s，重采样后形状：%s，文件名：%s' % (_raw_data.shape, raw_data.shape, filename))

    return raw_data


# ============== 数据处理部分 ==============


def yaml_read(yaml_path):
    """
    yaml配置读取器，
    输入yaml文件路径
    返回变量与值的字典
    """
    try:
        with open(yaml_path) as f:
            _info_yaml = f.read()
    except UnicodeDecodeError:
        # print('捕获编码问题，尝试中文编码')
        with open(yaml_path, encoding='utf-8') as f:
            _info_yaml = f.read()

    info_yaml = yaml.safe_load(_info_yaml)
    return info_yaml


def df_parser(x, mode=0, unit='s', date_format='%Y-%m-%d %H:%M:%S'):
    """
    pd.DataFrame读取时，时间解析格式
    pd.read_csv(data_filename, header=0,
                        index_col=0,  # 第一列读取为索引
                        parse_dates=[0],  # 第一列解析为时间
                        date_parser=df_parser  # 时间解析方法
                        )

    x为该行数据
    mode为解析格式，默认为自适应解析，先解析为以每秒为间隔，如果有问题再次尝试为以日期为解析格式

    """
    if mode == 0:
        # 模式0，将时间字符串先尝试解析为以秒为单位，后尝试解析为日期格式
        try:
            # 先尝试解析为以秒为单位
            z = pd.to_datetime(x, unit=unit)
        except Exception as e:
            try:
                # 后尝试解析为日期格式
                z = pd.to_datetime(x, format=date_format)
            except Exception as e:
                raise e


    elif mode == 1:
        # 模式1，将时间字符串解析为以秒为单位
        z = pd.to_datetime(x, unit=unit)
    elif mode == 2:
        # 模式2，将日期格式解析成指定格式
        z = pd.to_datetime(x, format=date_format)

    return z


def df2list(data, N, length, features, shift, step):
    """
    N = 滑窗个数 需要计算
    length = 滑窗长度 72
    features = 选择特征列 [0, 1, ...]
    shift = 向后偏移行数 length
    step = 滑动间隔 1
    return dy_list：返回滑窗 三维矩阵 [滑窗的个数，每个滑窗的长度，特征维度]
    """
    dy_list = []
    for i in range(N):
        dy = data[i * step + shift: i * step + shift + length, features]
        assert dy.shape[0] == length, '形状%s的长度小于length:%s' % (str(dy.shape), length)
        dy_list.append(dy)

    dy_list = np.array(dy_list)
    # print(dy_list.shape)
    return dy_list


def generate_xy(data, x_length=4, y_length=4, x_features=[0, 1], y_features=[2, 3], step=1, return_y_info=False):
    """
    将二维时序数据转为三维时间滑窗
    data：时序数据 二维矩阵 [时间，特征维度]
    x_length：输入样本和标签的长度，即输入m个时间点的数据
    y_length：输出样本和标签长度，即输出预测未来n个时间点的数据，int, >1
    x_features：特征维度中，样本列号，例如前2列是样本[0, 1]
    y_features：特征维度中，标签列号，例如后3列是标签[-3, -2, -1]
    step：每次滑窗滑动的步长，一般为1，int，>=1
    """

    y_shift = x_length
    x_shift = 0
    N = (len(data) - y_length - y_shift) // step + 1
    # step 余数 需要舍弃，可以舍弃左面或舍弃右面，并打印提示余数，默认舍弃左面
    remainder = (len(data) - y_length - y_shift) % step
    if remainder != 0:
        print('step取 {}时，存在余数remainder {}'.format(step, remainder))

    x = df2list(data=np.array(data), N=N, length=x_length, features=x_features, shift=x_shift + remainder, step=step)
    y = df2list(data=np.array(data), N=N, length=y_length, features=y_features, shift=y_shift + remainder, step=step)

    # 部分算法需要返回预测y窗口对应的x，这个x与输入的x不同，是输出y时刻将记录得到的x
    if return_y_info:
        y_x_info = df2list(data=np.array(data), N=N, length=y_length, features=x_features, shift=y_shift + remainder,
                           step=step)
        return x, y, y_x_info

    return x, y


def period_generate_xy(data_list, x_length, y_length, x_features, y_features, step=1):
    """
    流动周期列表的时间滑窗生成

    """
    x_list, y_list = [], []
    for data in data_list:
        x, y = generate_xy(data.values, x_length, y_length, x_features, y_features, step)

        x_list.append(x)
        y_list.append(y)

    x_list = np.vstack(x_list)
    y_list = np.vstack(y_list)

    return x_list, y_list


# ================================= 标准化方法 =================================

# 均值方差标准化
def raw2stand(raw_data, standardized_args, epsilon=1e-6):
    mean, std = standardized_args
    return (raw_data - mean) / (std + epsilon)


def stand2raw(stand_data, standardized_args, epsilon=1e-6):
    mean, std = standardized_args
    return stand_data * (std + epsilon) + mean


# EVN标准化方法
# def raw2stand(raw_data, mean, std, epsilon=1e-8):
#     return (raw_data - mean) / (mean + epsilon)

# def stand2raw(stand_data, mean, std, epsilon=1e-8):
#     return stand_data * (mean + epsilon) + mean


# 截断truncation log标准化
# def raw2stand(raw_data, *args):
#     new_raw_data = raw_data.copy()
#     _min = min(new_raw_data.min())
#     # assert _min > -100, '最小值超出正常情况:%s' % _min
#     if _min < -100:
#         print('最小值超出正常情况:%s' % _min)
#     new_raw_data[new_raw_data < 0] = 0
#     stand = np.log10(new_raw_data + 10)
#     return stand

# def stand2raw(stand_data, *args):
#     stand = np.power(10, stand_data) - 10
#     return stand


# ================================= 标准化方法 =================================


def series2time(series_tensor, step=1):
    """
    将series_tensor[n个时间窗口, time_step, feature]的时序数据三维张量
    变回为time_matrix[n+time_step-step, feature]原始数据长度二维矩阵

    step：构建时间滑窗时的滑动步长，一般为1
    """
    # 将第一个窗口数据完全保留
    # 从第二个开始的窗口数据只保留后step个
    # 如果step == time_step，则代表窗口每次移动了整个窗口大小，则全部保留
    added_matrix = np.vstack(series_tensor[1:, -step:, :])
    time_matrix = np.vstack([series_tensor[0], added_matrix])
    return time_matrix


def interrupt_time2period_list(raw_data, delta_time="1 hours"):
    """
    原始数据不存在缺失值，大多数时间点之间连续，
    部分时间段出现长时间的跳过，少部分时间点的间隔出现不规律
    每一段连续存在数据的部分作为一个周期截取下来单独保存

    # TODO 增加缺失值的处理功能，比如删除或插值

    raw_data：原始存在时间断层的数据
    delta_time：最长时间间隔，默认1小时
    return period_resample_raw_data_list 将不同流动周期数据拆分成长度不同的元素，
            再组成列表，每个元素都是二维矩阵[时间，特征维度]
    """

    # 连续时间索引
    time_index_continuous = raw_data.index
    # 连续时间索引进行差分
    time_index_continuous_diff = np.ediff1d(time_index_continuous)

    # 每个流动周期的最长时间间隔为1小时
    # 即超过1小时的时间空白认为不同的流动周期
    # 小于一小时则认为是同一个流动周期

    # 每两个时间点的差大于一小时的位置为流动周期 delta_time = "1 hours"
    interval_flag = time_index_continuous_diff > pd.Timedelta(delta_time)
    # 流动周期数据量
    num_period = interval_flag.sum()
    # 流动周期行号
    period_row_list = np.where(interval_flag)[0]

    # 将原始数据分为流动周期的原始数据 period_raw_data_list
    period_raw_data_list = []
    n0 = 0
    n1 = period_row_list[0]
    for n in range(num_period + 1):
        # 从原始数据中截取到每个流动周期的数据
        period_raw_data = raw_data[n0 + 1: n1 + 1]
        period_raw_data_list.append(period_raw_data)

        n0 = n1
        if (n + 1) >= num_period:
            n1 = len(time_index_continuous_diff)
            continue
        n1 = period_row_list[n + 1]

    period_resample_raw_data_list = []
    for period_raw_data in period_raw_data_list:
        period_resample_raw_data = period_raw_data.resample('T').mean()
        period_resample_raw_data = period_resample_raw_data.interpolate(method='linear')

        period_resample_raw_data_list.append(period_resample_raw_data)

    return period_resample_raw_data_list


# ============== matplotlib画图 ==============


def draw1(gt, pred, is_show=False, img_path=None, title=[], show_error=True):
    """
    gt 真实值
    pred 预测值

    title：每列数据的标题名
    """
    s = 0.5
    fig = plt.figure(figsize=(15, 3))
    for i in range(gt.shape[1]):
        plt.subplot(1, gt.shape[1], i + 1)
        x = range(len(gt[:, i]))
        plt.scatter(x, gt[:, i], label='Truth', s=s)
        plt.scatter(x, pred[:, i], label='Predict', s=s)
        if show_error:
            plt.scatter(x, gt[:, i] - pred[:, i], label='Error', c='g', s=s)
        if len(title) == gt.shape[1]:
            temp_title = list(title[i])
            if len(temp_title) > 10:
                # temp_title.insert(10, '\n')
                temp_title = temp_title[:20]
            temp_title = ''.join(temp_title)
            plt.title(temp_title)
        # plt.legend()

    plt.tight_layout()

    if img_path != None:
        makedirs(img_path)
        plt.savefig(img_path, bbox_inches='tight', dpi=300)
    if is_show:
        plt.show()
    else:
        plt.close(fig)


def draw_PT_Qowg(PT_Qowg_data):
    """
    此函数专门用于绘制压力、温度、油气水流量，即二维矩阵[时间点，特征]
    PT_Qowg_data:即二维矩阵

    暂时没有返回和保存功能
    """

    index_plt = PT_Qowg_data.index

    # 画输入 PT 压力温度 图像
    fig, ax1 = plt.subplots()  # figsize=(10, 6)
    ax1.scatter(index_plt, PT_Qowg_data.iloc[:, 0], c='g', s=1)
    plt.xticks(rotation=30)
    plt.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.scatter(index_plt, PT_Qowg_data.iloc[:, 1], c='r', s=1)

    plt.xlim([index_plt[0], index_plt[-1]])
    # fig.legend(loc=(0.12, 0.8))
    plt.legend(loc='upper right')

    plt.show()

    # 画输出 Qo Qw Qg  油气水流量 图像
    fig, ax1 = plt.subplots()  # figsize=(10, 6)
    ax1.scatter(index_plt, PT_Qowg_data.iloc[:, 2], c='g', s=1)
    ax1.scatter(index_plt, PT_Qowg_data.iloc[:, 3], c='b', s=1)
    plt.xticks(rotation=30)
    plt.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.scatter(index_plt, PT_Qowg_data.iloc[:, 4], c='r', s=1)

    plt.xlim([index_plt[0], index_plt[-1]])
    # fig.legend(loc=(0.09, 0.75))
    plt.legend(loc='upper right')

    plt.show()
