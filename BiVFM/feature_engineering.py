#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/28 11:27
# @Author  : 马赫
# @Email   : 1692303843@qq.com
# @FileName: feature_engineering.py.py
# 特征工程  feature_engineering(raw_data)


def feature_engineering(raw_data):
    """
    特征工程

    输入与输出都是pd.DataFrame

    """

    new_data = raw_data.copy()

    # 构建压力差值作为新特征

    new_data['P_choke_diff_a'] = raw_data['Pressure head_a'] - raw_data['Pressure choke_a']
    new_data['P_choke_ratio_a'] = raw_data['Pressure head_a'] / raw_data['Pressure choke_a']
    new_data['P_choke_sum_diff_a'] = raw_data['Pressure choke_a'] - raw_data['Pressure outlet']

    new_data['P_choke_diff_b'] = raw_data['Pressure head_b'] - raw_data['Pressure choke_b']
    new_data['P_choke_ratio_b'] = raw_data['Pressure head_b'] / raw_data['Pressure choke_b']
    new_data['P_choke_sum_diff_b'] = raw_data['Pressure choke_b'] - raw_data['Pressure outlet']

    new_data['P_choke_diff_c'] = raw_data['Pressure head_c'] - raw_data['Pressure choke_c']
    new_data['P_choke_ratio_c'] = raw_data['Pressure head_c'] / raw_data['Pressure choke_c']
    new_data['P_choke_sum_diff_c'] = raw_data['Pressure choke_c'] - raw_data['Pressure outlet']

    # 构建温度差值作为新特征

    new_data['TM_choke_diff_a'] = raw_data['TM head_a'] - raw_data['TM choke_a']
    new_data['TM_choke_ratio_a'] = raw_data['TM head_a'] / raw_data['TM choke_a']
    new_data['TM_choke_sum_diff_a'] = raw_data['TM choke_a'] - raw_data['TM outlet']

    new_data['TM_choke_diff_b'] = raw_data['TM head_b'] - raw_data['TM choke_b']
    new_data['TM_choke_ratio_b'] = raw_data['TM head_b'] / raw_data['TM choke_b']
    new_data['TM_choke_sum_diff_b'] = raw_data['TM choke_b'] - raw_data['TM outlet']

    new_data['TM_choke_diff_c'] = raw_data['TM head_c'] - raw_data['TM choke_c']
    new_data['TM_choke_ratio_c'] = raw_data['TM head_c'] / raw_data['TM choke_c']
    new_data['TM_choke_sum_diff_c'] = raw_data['TM choke_c'] - raw_data['TM outlet']

    # 构建总流量差值和变化率
    new_data['QT_diff of change'] = raw_data['QT'].diff().fillna(0)

    new_data['QT_rate of change'] = raw_data['QT']
    new_data['QT_rate of change'][1:] = raw_data['QT'][1:] / raw_data['QT'][:-1].values
    new_data['QT_rate of change'][0] = 1

    anew_columns_order = [

        'Pressure head_a',
        'Pressure choke_a',
        'Pressure head_b',
        'Pressure choke_b',
        'Pressure head_c',
        'Pressure choke_c',
        'Pressure outlet',

        # x: P variant
        'P_choke_diff_a',
        'P_choke_ratio_a',
        'P_choke_sum_diff_a',
        'P_choke_diff_b',
        'P_choke_ratio_b',
        'P_choke_sum_diff_b',
        'P_choke_diff_c',
        'P_choke_ratio_c',
        'P_choke_sum_diff_c',

        'TM head_a',
        'TM choke_a',
        'TM head_b',
        'TM choke_b',
        'TM head_c',
        'TM choke_c',
        'TM outlet',

        # x: TM variant
        'TM_choke_diff_a',
        'TM_choke_ratio_a',
        'TM_choke_sum_diff_a',
        'TM_choke_diff_b',
        'TM_choke_ratio_b',
        'TM_choke_sum_diff_b',
        'TM_choke_diff_c',
        'TM_choke_ratio_c',
        'TM_choke_sum_diff_c',

        # x: QT and variant
        'QT_diff of change',
        'QT_rate of change',

        'QT',

        # y
        'QL',
        'QO',
        'QW',
        'QG',
    ]
    new_data = new_data.loc[:, anew_columns_order]

    return new_data
