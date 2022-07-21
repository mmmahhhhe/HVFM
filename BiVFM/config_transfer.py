#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/6/7 11:46
# @Author  : 马赫
# @Email   : 1692303843@qq.com
# @FileName: config_transfer.py


try:
    from BiVFM.func import *
except Exception:
    from func import *


def get_base_config(base_yaml_path):
    """
    get config from yaml and update

    base_yaml_path = r'xxx\demo_work1\conf_0_base.yaml'
    base_conf = get_base_config(base_yaml_path)
    for k, v in base_conf.items():
        print('%s:\t%s' % (k, v))

    :param base_yaml_path: str config file path
    :return: dict of config
    """

    def path_join(*args):
        return '/'.join(args)

    base_conf = yaml_read(base_yaml_path)

    # =============== base config update ===============
    base_conf['learning_rate'] = float(base_conf['learning_rate'])


    base_conf['autoolga_output_dir'] = path_join(base_conf['base_dir'], base_conf['autoolga_output_dir'])

    base_conf['extract_xy'] = path_join(base_conf['base_dir'], base_conf['extract_xy'])

    base_conf['tpl_dir'] = path_join(base_conf['base_dir'], base_conf['tpl_dir'])
    base_conf['raw_csv_dir'] = path_join(base_conf['base_dir'], base_conf['raw_csv_dir'])
    base_conf['xy_dir'] = path_join(base_conf['base_dir'], base_conf['xy_dir'])
    base_conf['pkl_dir'] = path_join(base_conf['base_dir'], base_conf['pkl_dir'])

    base_conf['mean_path'] = path_join(base_conf['base_dir'], base_conf['mean_path'])
    base_conf['std_path'] = path_join(base_conf['base_dir'], base_conf['std_path'])


    base_conf['train_log_file'] = path_join(base_conf['base_dir'], base_conf['info_path'], base_conf['train_log_file'])
    base_conf['other_log_file'] = path_join(base_conf['base_dir'], base_conf['info_path'], base_conf['other_log_file'])

    base_conf['save_path'] = path_join(base_conf['base_dir'], base_conf['info_path'], base_conf['save_path'])
    base_conf['img_save_path'] = path_join(base_conf['base_dir'], base_conf['info_path'], base_conf['img_save_path'])

    base_conf['opt_path'] = path_join(base_conf['base_dir'], base_conf['info_path'], base_conf['opt_path'])
    base_conf['model_path'] = path_join(base_conf['base_dir'], base_conf['info_path'], base_conf['model_path'])

    return base_conf


if __name__ == '__main__':
    base_yaml_path = r'D:\work\VFM\BiVFM\program\demo_work1\conf_0_base.yaml'
    base_conf = get_base_config(base_yaml_path)
    for k, v in base_conf.items():
        print('%s\t%s\t%s' % (k, type(v), v))



