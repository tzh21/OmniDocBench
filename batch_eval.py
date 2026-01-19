#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量评估脚本（支持并行）
遍历 saved_outputs 下的所有目录，找到 *_md 子目录进行评估
"""

import os
import sys
import yaml
import io
import pathlib
import argparse
import copy
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count


def find_md_subdir(parent_dir):
    """查找目录下的 *_md 子目录"""
    for item in os.listdir(parent_dir):
        item_path = os.path.join(parent_dir, item)
        if os.path.isdir(item_path) and item.endswith('_md'):
            return item_path
    return None


def run_single_evaluation(args_tuple):
    """
    运行单个评估任务（用于并行）
    args_tuple: (dir_name, md_dir, config_path)
    """
    dir_name, md_dir, config_path = args_tuple
    
    # 在子进程中重新导入模块
    from registry.registry import EVAL_TASK_REGISTRY, DATASET_REGISTRY, METRIC_REGISTRY
    import dataset
    import task
    import metrics
    
    try:
        # 加载配置
        with io.open(os.path.abspath(config_path), "r", encoding="utf-8") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        
        for task_name in config.keys():
            if not config.get(task_name):
                continue
                
            dataset_name = config[task_name]['dataset']['dataset_name']
            metrics_list = config[task_name]['metrics']
            
            # 更新预测路径
            config[task_name]['dataset']['prediction']['data_path'] = md_dir
            
            val_dataset = DATASET_REGISTRY.get(dataset_name)(config[task_name])
            val_task = EVAL_TASK_REGISTRY.get(task_name)
            
            save_name = os.path.basename(md_dir) + '_' + config[task_name]['dataset'].get('match_method', 'quick_match')
            print(f'[{dir_name}] Processing: {save_name}')
            
            if config[task_name]['dataset']['ground_truth'].get('page_info'):
                val_task(val_dataset, metrics_list, config[task_name]['dataset']['ground_truth']['page_info'], save_name)
            else:
                val_task(val_dataset, metrics_list, config[task_name]['dataset']['ground_truth']['data_path'], save_name)
        
        return (dir_name, "成功", None)
    
    except Exception as e:
        import traceback
        return (dir_name, "失败", str(e))


def main():
    parser = argparse.ArgumentParser(description='批量评估脚本（支持并行）')
    parser.add_argument('--saved_outputs', type=str, 
                        default='/export/home/tangzihan.15/fast-ocr/local/saved_outputs',
                        help='saved_outputs 目录路径')
    parser.add_argument('--config', '-c', type=str, 
                        default='./configs/end2end.yaml',
                        help='基础配置文件路径')
    parser.add_argument('--filter', type=str, default=None,
                        help='过滤目录名，只评估包含该字符串的目录')
    parser.add_argument('--workers', '-w', type=int, default=8,
                        help='并行worker数量')
    parser.add_argument('--sequential', action='store_true',
                        help='顺序执行（不并行）')
    args = parser.parse_args()
    
    # 获取所有需要评估的目录
    saved_outputs_dir = args.saved_outputs
    all_dirs = sorted([d for d in os.listdir(saved_outputs_dir) 
                       if os.path.isdir(os.path.join(saved_outputs_dir, d))])
    
    # 应用过滤器
    if args.filter:
        all_dirs = [d for d in all_dirs if args.filter in d]
    
    # 准备任务列表
    tasks = []
    for dir_name in all_dirs:
        dir_path = os.path.join(saved_outputs_dir, dir_name)
        md_dir = find_md_subdir(dir_path)
        
        if md_dir is None:
            print(f"跳过 {dir_name}: 未找到 *_md 子目录")
            continue
        
        tasks.append((dir_name, md_dir, os.path.abspath(args.config)))
    
    print(f"找到 {len(tasks)} 个目录需要评估")
    print(f"并行 workers: {args.workers if not args.sequential else 1}")
    print("=" * 60)
    
    results_summary = []
    
    if args.sequential:
        # 顺序执行
        for task_args in tasks:
            print(f"\n正在评估: {task_args[0]}")
            result = run_single_evaluation(task_args)
            results_summary.append(result)
    else:
        # 并行执行
        num_workers = min(args.workers, len(tasks), cpu_count())
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_task = {executor.submit(run_single_evaluation, t): t[0] for t in tasks}
            
            for future in as_completed(future_to_task):
                dir_name = future_to_task[future]
                try:
                    result = future.result()
                    results_summary.append(result)
                    status = "✓" if result[1] == "成功" else "✗"
                    print(f"[{status}] 完成: {dir_name}")
                except Exception as e:
                    results_summary.append((dir_name, "失败", str(e)))
                    print(f"[✗] 失败: {dir_name} - {e}")
    
    # 打印汇总
    print("\n" + "=" * 60)
    print("批量评估完成！汇总：")
    print("=" * 60)
    
    success_count = 0
    fail_count = 0
    
    for name, status, error in results_summary:
        if status == "成功":
            print(f"  ✓ {name}")
            success_count += 1
        else:
            print(f"  ✗ {name}: {error}")
            fail_count += 1
    
    print(f"\n成功: {success_count}, 失败: {fail_count}")
    print(f"结果保存在 ./result/ 目录下")


if __name__ == '__main__':
    main()
