#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/28 14:11
# @Author  : 马赫
# @Email   : 1692303843@qq.com
# @FileName: olga_func.py


import re
import os
import pandas as pd
import matplotlib.pyplot as plt
try:
    from BiVFM.func import *
except Exception:
    from func import *


def olga_columns_info_parse(olga_columns_type_list):
    """
    OLGA列名信息拆分
    输入列名列表
    返回列名拆分列表

    示例输入
    ["VOLGBL 'GLOBAL' '(-)' 'Global max volume error since last write'",
    "HT 'GLOBAL' '(S)' 'Time step'",
    "GL 'BOUNDARY:' 'BRANCH:' 'PIPELINE' 'PIPE:' 'p2' 'NR:' '2'  '(KG/S)' 'Liquid bulk mass flow'",
    "ID 'BOUNDARY:' 'BRANCH:' 'PIPELINE' 'PIPE:' 'p2' 'NR:' '2'  '(-)' 'Flow regime: 1=Stratified, 2=Annular, 3=Slug, 4=Bubble.'",
    "PT 'SECTION:' 'BRANCH:' 'PIPELINE' 'PIPE:' 'p2' 'NR:' '2'  '(PA)' 'Pressure'",
    "LIQC 'BRANCH:' 'PIPELINE' '(M3)' 'Total liquid content in branch'",
    "CGGBOU 'NODE:' 'OUTLET' '(KG/S)' 'Gas mass flow for each component at boundary node'",
    "GLTHLBOU 'NODE:' 'OUTLET' '(KG/S)' 'Oil mass flow at boundary node'",
    "GLTWTBOU 'NODE:' 'OUTLET' '(KG/S)' 'Water mass flow at boundary node'",
    "GTBOU 'NODE:' 'OUTLET' '(KG/S)' 'Total mass flow at boundary node'",
    "PT 'NODE:' 'OUTLET' '(PA)' 'Pressure'",
    "PTBOU 'NODE:' 'OUTLET' '(PA)' 'Pressure at boundary node'"]


    示例返回

    [['VOLGBL', '', '', 'Global max volume error since last write'],
    ['HT', 'S', '', 'Time step'],
    ['GL', 'KG/S', 'PIPELINE.p2.2', 'Liquid bulk mass flow'],
    ['ID',
    '',
    'PIPELINE.p2.2',
    'Flow regime: 1=Stratified, 2=Annular, 3=Slug, 4=Bubble.'],
    ['PT', 'BAR', 'PIPELINE.p2.2', 'Pressure'],
    ['LIQC', 'M3', 'PIPELINE', 'Total liquid content in branch'],
    ['CGGBOU',
    'KG/S',
    'OUTLET',
    'Gas mass flow for each component at boundary node'],
    ['GLTHLBOU', 'KG/S', 'OUTLET', 'Oil mass flow at boundary node'],
    ['GLTWTBOU', 'KG/S', 'OUTLET', 'Water mass flow at boundary node'],
    ['GTBOU', 'KG/S', 'OUTLET', 'Total mass flow at boundary node'],
    ['PT', 'BAR', 'OUTLET', 'Pressure'],
    ['PTBOU', 'BAR', 'OUTLET', 'Pressure at boundary node']]
    """

    olga_columns_type_split_list = []
    for olga_columns_type_i in olga_columns_type_list:
        _key = re.findall("(.*?) '.*", olga_columns_type_i)[0]
        _describe = re.findall("'(.*?)'", olga_columns_type_i)

        unit = _describe[-2][1:-1]
        if _describe[-2][1:-1] == '-':
            unit = ''

        position = ''
        for _name in _describe[1: -2]:
            if _name in ['BRANCH:', 'PIPE:', 'NR:']:
                if _name in ['PIPE:', 'NR:']:
                    position += '.'
                continue
            position += _name
        describe = _describe[-1]
        # 关键字 [单位] (位置) 解释
        # key = '%s\t[%s]\t(%s)\t"%s"' % (_key, unit, position, describe)
        columns_info = [_key, unit, position, describe]
        olga_columns_type_split_list.append(columns_info)
    return olga_columns_type_split_list


def trend_plot_data(olga_time_plot_path):
    """
    从OLGA时间趋势数据文件中提取时间数据
    olga_time_plot_path: 'xxx.tpl'文件
    """
    with open(olga_time_plot_path) as f:
        str_data = f.readlines()
    str_data = [_.strip() for _ in str_data]

    # 从列名处切分，前面是olga信息，后面是列数据
    olga_info_lines_id = str_data.index("CATALOG")
    # 一共统计多少列变量
    olga_columns_num = int(str_data[olga_info_lines_id + 1])
    olga_columns_type = str_data[olga_info_lines_id + 2:olga_info_lines_id + 2 + olga_columns_num]

    # 将获取到的列名进行切分
    olga_columns_type_dict = olga_columns_info_parse(olga_columns_type)
    olga_data_df = str_data[olga_info_lines_id + 2 + olga_columns_num + 1:]

    olga_data_df = pd.DataFrame([_.split() for _ in olga_data_df], dtype='float')

    # olga_data_df[0] = pd.to_datetime(olga_data_df[0], unit='s')
    olga_data_df.set_index(0, inplace=True)
    olga_data_df.index.name = 'Time(sec)'
    olga_data_df.columns = ['%s[%s](%s)"%s"' % tuple(info) for info in olga_columns_type_dict]

    return olga_data_df


def profile_plot_data(olga_profile_plot_path):
    """
    从OLGA时间剖面数据文件中提取剖面数据
    olga_profile_plot_path: 'xxx.ppl'文件
    """
    with open(olga_profile_plot_path) as f:
        str_data = f.readlines()
    str_data = [_.strip() for _ in str_data]

    # 从列名处切分，前面是olga信息，后面是列数据
    olga_info_lines_id = str_data.index("CATALOG")

    # 获取沿程长度
    olga_info_x_alone_id = str_data[:olga_info_lines_id].index("'PIPELINE'")
    _olga_info_x_alone = str_data[olga_info_x_alone_id + 2: olga_info_x_alone_id + 2 + (
                olga_info_lines_id - olga_info_x_alone_id) // 2 - 1]
    olga_info_x_alone = []
    for _ in _olga_info_x_alone:
        olga_info_x_alone += _.split(' ')
    olga_info_x_alone = [float('%.3g' % float(i)) for i in olga_info_x_alone]

    # 一共统计多少列变量
    olga_columns_num = int(str_data[olga_info_lines_id + 1])
    olga_columns_type = str_data[olga_info_lines_id + 2:olga_info_lines_id + 2 + olga_columns_num]

    # 将获取到的列名进行切分
    olga_columns_type_dict = olga_columns_info_parse(olga_columns_type)
    olga_data_list = str_data[olga_info_lines_id + 2 + olga_columns_num + 1:]
    olga_data_list = [i.split(' ') for i in olga_data_list]

    # 分离出每个时间点的数据
    values_list = []
    values = []
    time_step = float(olga_data_list[0][0])
    for i in olga_data_list:
        if len(i) == 1:
            if len(values) == 0:
                continue
            values_list.append([time_step, values])
            values = []
            time_step = float(i[0])
        else:
            values.append(i)
    values_list.append([time_step, values])

    # 每个时间点都是一个pandas
    """
    注意！！！
    这里如果有的沿程坐标缺少某属性的值，则会把缺失值填pandas.NaN
    """
    olga_data_time_steps = []
    for i in values_list:
        time_step = i[0]
        values = i[1]
        values = pd.DataFrame(values, dtype='float').T
        values.columns = ['%s[%s](%s)"%s"' % tuple(info) for info in olga_columns_type_dict]
        values.index = olga_info_x_alone
        olga_data_time_steps.append([time_step, values])

    return olga_data_time_steps


def olga_trend2csv(olga_time_plot_path, save_path, is_show=False):
    """
    将OLGA生成的tpl文件转为csv文件
    """
    # olga_time_plot_path = 'olga_work1/q4.tpl' # 时间节点信息
    # 趋势数据
    olga_trend_data = trend_plot_data(olga_time_plot_path)

    if is_show:
        for i in range(olga_trend_data.shape[1]):
            olga_trend_data.iloc[:, i].plot(legend=False)
            plt.show()

    # 将趋势数据保存为csv文件
    makedirs(save_path)
    olga_trend_data.to_csv(save_path)


def tpl_list2raw(tpl_dir, raw_csv_dir):
    """
    将tpl文件夹转为csv文件夹
    """
    rm_dir(raw_csv_dir)
    makedirs(raw_csv_dir)

    for i in os.listdir(tpl_dir):
        # 避免`.ipynb_checkpoints`文件
        if not i.endswith('.tpl'):
            continue
        # 将OLGA的.tpl文件转为cvs格式
        _old_path = '%s/%s' % (tpl_dir, i)
        _new_path = '%s/%s.csv' % (raw_csv_dir, os.path.splitext(i)[0])

        olga_trend2csv(olga_time_plot_path=_old_path, save_path=_new_path, is_show=False)


def featrue_extract(path_data, path_save, extract_x, extract_y):
    """
    将OLGA提取出的数据选择其中需要的部分，前几列为x，后面的其余列为y
    extract_x = [
                    [11],  # 保存的列号
                    ['Pressure outlet'],  # 修改后保存的列名
                    [0.00001],  # 单位变化多少倍。 压力单位从Pa变为bar
                    ['PTBOU[PA](OUTLET)"Pressure at boundary node"']  # 修改前的列名
                ]

    extract_y = [
                    [7, 6],
                    ['Oil mass', 'Gas mass'],
                    [1, 1],  # 单位变化多少倍。无变化
                    [
                    'GLTHLBOU[KG/S](OUTLET)"Oil mass flow at boundary node"',
                    'CGGBOU[KG/S](OUTLET)"Gas mass flow for each component at boundary node"'
                    ]
                ]
    featrue_extract(
                    path_data='olga_work1/dataset0/my_1_slug.csv',
                    path_save='olga_work1/dataset1/my_1_slug.csv',
                    extract_x=extract_x, extract_y=extract_y)
    """

    _raw_data = pd.read_csv(path_data, header=0,
                            index_col=0,
                            parse_dates=[0],
                            date_parser=lambda x: df_parser(x, mode=0, unit='s', date_format='%Y-%m-%d %H:%M:%S'),
                            engine='python'
                            )

    # 截取xy参数
    part_df = _raw_data.iloc[:, extract_x[0] + extract_y[0]]
    # 乘单位换算系数
    part_df = part_df * (extract_x[2] + extract_y[2])
    # print('Origin column names:')
    # for i in part_df.columns:
    #     print('\t', i)
    assert part_df.columns.tolist() == extract_x[3] + extract_y[3], \
        '原始列名跟修改后的列名不一致，请检查\n%s\n%s' % (part_df.columns.tolist(), extract_x[3] + extract_y[3])
    part_df.columns = extract_x[1] + extract_y[1]
    # print('Extract column names:')
    # for i in part_df.columns:
    #     print('\t', i)
    part_df.to_csv(path_save)


def featrue_extract_dir(raw_csv_dir, xy_csv_dir, extract_x, extract_y):
    """
    `featrue_extract`函数的文件夹批量模式
    将文件夹1提取指定的xy到文件夹2

    """
    rm_dir(xy_csv_dir)
    makedirs(xy_csv_dir)

    for i in os.listdir(raw_csv_dir):
        # 避免临时缓存文件影响
        if i == '.ipynb_checkpoints':
            continue
        old = '%s/%s' % (raw_csv_dir, i)
        new = '%s/%s' % (xy_csv_dir, i)

        featrue_extract(
            path_data=old,
            path_save=new,
            extract_x=extract_x,
            extract_y=extract_y
        )





