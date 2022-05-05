from multiprocessing import freeze_support

from common.olga_base import *
from common.utils import *
from common.core_regis import *
from fleet.olga_master import *


# ======================== 多任务生成 ========================
def prepare(distributed, origin_path, migration_base_path):

    olga_generate(origin_path, migration_base_path)

    migration_path_list = ['%s/%s' % (migration_base_path, i) for i in os.listdir(migration_base_path)]


    log('============ 全部待运行OLGA项目：%s个 ============' % len(migration_path_list), log_file='%s/log.txt' % origin_path)
    log(migration_path_list, log_file='%s/log.txt' % origin_path)
    log('============ 开始运行... ============', log_file='%s/log.txt' % origin_path)
    return migration_path_list


# ======================== 多任务执行 ========================
def main():
    CORE_PATH = 'numpy/core/OlgaExecutables/OLGA-2015.1.2.exe'
    CORE_PATH = core_regis(CORE_PATH)

    distributed = config['distributed']

    origin_path = config['origin_path']
    migration_base_path = config['migration_base_path']

    # 多任务生成
    migration_path_list = prepare(distributed, origin_path, migration_base_path)

    print(migration_path_list)
    # 单机运行
    if not distributed:
        alone_run(origin_path, migration_path_list, CORE_PATH)

    # 分布式运行
    else:

        manager.start()
        manager_master(manager, migration_base_path)


if __name__ == '__main__':
    freeze_support()
    main()