#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量评估脚本（支持并行）
遍历 saved_outputs 下的所有目录，找到固定子目录 ocr_results_md 进行评估，
结果 JSON 以一级目录名命名。
"""

import os
import re
import shutil
import json

# 预测结果子目录固定名称
MD_SUBDIR_NAME = 'ocr_results_md'
import sys
import yaml
import io
import pathlib
import argparse
import copy
from glob import glob

import numpy as np
import pandas as pd

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count


def move_timestamp_to_end(name):
    """若名称以 MMDD-HHMMSS_ 开头，返回 (新名称, True)；否则返回 (原名称, False)"""
    m = re.match(r'^(\d{4}-\d{6})_(.+)$', name)
    if m:
        return f"{m.group(2)}_{m.group(1)}", True
    return name, False


def find_md_subdir(parent_dir, subdir_name=MD_SUBDIR_NAME):
    """查找目录下指定名称的子目录（默认 ocr_results_md）"""
    md_path = os.path.join(parent_dir, subdir_name)
    return md_path if os.path.isdir(md_path) else None


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
            
            # 结果 JSON 以一级目录名（dir_name）命名
            save_name = dir_name + '_' + config[task_name]['dataset'].get('match_method', 'quick_match')
            print(f'[{dir_name}] Processing: {save_name}')
            
            if config[task_name]['dataset']['ground_truth'].get('page_info'):
                val_task(val_dataset, metrics_list, config[task_name]['dataset']['ground_truth']['page_info'], save_name)
            else:
                val_task(val_dataset, metrics_list, config[task_name]['dataset']['ground_truth']['data_path'], save_name)
        
        return (dir_name, "成功", None)
    
    except Exception as e:
        import traceback
        return (dir_name, "失败", str(e))


def load_match_method_from_config(config_path):
    """从 YAML 中取第一个启用任务的 match_method（与评测命名一致）。"""
    with io.open(os.path.abspath(config_path), "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for task_name in config.keys():
        if not config.get(task_name):
            continue
        return config[task_name]["dataset"].get("match_method", "quick_match")
    return "quick_match"


def extract_result_prefixes(result_folder, match_name):
    """从 result 目录下的文件名中提取前缀（与 generate_result_tables.ipynb 一致）。"""
    if not os.path.isdir(result_folder):
        return []
    prefixes = set()
    pattern = re.compile(rf"^(.+?)_{re.escape(match_name)}_.*\.json$")
    for filename in os.listdir(result_folder):
        m = pattern.match(filename)
        if m:
            prefixes.add(m.group(1))
    return sorted(prefixes)


def is_nan_like(v):
    if pd.isna(v):
        return True
    if isinstance(v, str) and str(v).strip().upper() in ("NAN", "N/A", "NA"):
        return True
    return False


def build_omni_score_markdown(result_folder, match_name):
    """
    汇总各 run 的 metric JSON，生成与 tools/generate_result_tables.ipynb 中
    overall 段相同的指标表（Markdown）。
    """
    prefix_list = extract_result_prefixes(result_folder, match_name)
    if not prefix_list:
        return (
            f"未在 `{result_folder}` 下找到匹配 `{match_name}` 的 `*_metric_result.json`，"
            "未生成指标表。\n"
        )

    dict_list = []
    for ocr_type in prefix_list:
        result_path = os.path.join(
            result_folder, f"{ocr_type}_{match_name}_metric_result.json"
        )
        with open(result_path, "r", encoding="utf-8") as f:
            result = json.load(f)

        save_dict = {}
        for category_type, metric in [
            ("text_block", "Edit_dist"),
            ("display_formula", "Edit_dist"),
            ("table", "TEDS"),
            ("reading_order", "Edit_dist"),
        ]:
            if metric in ("TEDS", "TEDS_structure_only"):
                if result[category_type]["page"].get(metric):
                    save_dict[category_type + "_" + metric] = (
                        result[category_type]["page"][metric]["ALL"] * 100
                    )
                else:
                    save_dict[category_type + "_" + metric] = 0
            else:
                save_dict[category_type + "_" + metric] = result[category_type][
                    "all"
                ][metric].get("ALL_page_avg", np.nan)

        dict_list.append(save_dict)

    df = pd.DataFrame(dict_list, index=prefix_list)

    overall_cols = [
        "text_block_Edit_dist",
        "display_formula_Edit_dist",
        "table_TEDS",
        "reading_order_Edit_dist",
    ]
    nan_reports = []
    for col in overall_cols:
        if col not in df.columns:
            continue
        for idx in df.index:
            v = df.loc[idx, col]
            if is_nan_like(v):
                nan_reports.append((idx, col, repr(v)))

    lines = []
    if nan_reports:
        lines.append("【存在 NaN 的指标与数据集】\n")
        for dataset, col, raw in nan_reports:
            lines.append(f"- 数据集: `{dataset}`  列: `{col}`  原始值: {raw}\n")
        lines.append("\n")

    for col in overall_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.round(3)
    df["overall"] = (
        (1 - df["text_block_Edit_dist"]) * 100
        + (1 - df["display_formula_Edit_dist"]) * 100
        + df["table_TEDS"]
        + (1 - df["reading_order_Edit_dist"]) * 100
    ) / 4

    lines.append(df.to_markdown())
    lines.append("\n")
    return "".join(lines)


def write_omni_score_md(result_folder, saved_outputs_dir, config_path):
    """将 OmniDocBench overall 表写入 saved_outputs 目录下的 omni_scores.md。"""
    match_name = load_match_method_from_config(config_path)
    md_body = build_omni_score_markdown(result_folder, match_name)
    out_path = os.path.join(saved_outputs_dir, "omni_scores.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md_body)
    return out_path


def main():
    parser = argparse.ArgumentParser(description='批量评估脚本（支持并行）')
    parser.add_argument('saved_outputs', type=str,
                        help='saved_outputs 目录路径')
    parser.add_argument('--config', '-c', type=str, 
                        default='./configs/fastocr_end2end.yaml',
                        help='基础配置文件路径')
    parser.add_argument('--filters', '-f', type=str, nargs='+', default=None,
                        help='过滤目录名，只评估包含任一字符串的目录（取并集）')
    parser.add_argument('--workers', '-w', type=int, default=8,
                        help='并行worker数量')
    parser.add_argument('--sequential', action='store_true',
                        help='顺序执行（不并行）')
    parser.add_argument('--merge', action='store_true',
                        help='合并模式：不清空 result 目录，并跳过 result 中已有结果的数据点')
    args = parser.parse_args()

    result_dir = './result'
    if args.merge:
        # 合并模式：不清空 result，仅确保目录存在
        os.makedirs(result_dir, exist_ok=True)
    else:
        # 测试前清空 result 目录，再确保目录存在
        if os.path.isdir(result_dir):
            shutil.rmtree(result_dir)
        os.makedirs(result_dir, exist_ok=True)

    # 获取所有需要评估的目录
    saved_outputs_dir = os.path.expanduser(args.saved_outputs)
    # 先将名称以时间戳开头的子目录重命名，把时间戳移到最后（MMDD-HHMMSS_xxx -> xxx_MMDD-HHMMSS）
    for d in list(os.listdir(saved_outputs_dir)):
        full_path = os.path.join(saved_outputs_dir, d)
        if not os.path.isdir(full_path):
            continue
        new_name, need_rename = move_timestamp_to_end(d)
        if need_rename and new_name != d:
            new_path = os.path.join(saved_outputs_dir, new_name)
            if not os.path.exists(new_path):
                os.rename(full_path, new_path)
                print(f"重命名: {d} -> {new_name}")
            else:
                print(f"跳过重命名 {d}: 目标 {new_name} 已存在")
    all_dirs = sorted([d for d in os.listdir(saved_outputs_dir)
                       if os.path.isdir(os.path.join(saved_outputs_dir, d))])

    # 应用过滤器
    if args.filters:
        all_dirs = [d for d in all_dirs if any(filter_str in d for filter_str in args.filters)]

    # 合并模式下：预加载 config，用于判断某目录的预期结果文件是否已存在
    expected_save_names_per_dir = None
    if args.merge:
        with io.open(os.path.abspath(args.config), "r", encoding="utf-8") as f:
            merge_config = yaml.load(f, Loader=yaml.FullLoader)
        expected_save_names_per_dir = {}
        for task_name in merge_config.keys():
            if not merge_config.get(task_name):
                continue
            match_method = merge_config[task_name]['dataset'].get('match_method', 'quick_match')
            # 仅用于判断“某 dir 会生成哪些 save_name”，不依赖具体 dir，只记后缀
            expected_save_names_per_dir[task_name] = match_method

    # 准备任务列表
    tasks = []
    for dir_name in all_dirs:
        dir_path = os.path.join(saved_outputs_dir, dir_name)
        md_dir = find_md_subdir(dir_path)

        if md_dir is None:
            print(f"跳过 {dir_name}: 未找到 {MD_SUBDIR_NAME} 子目录")
            continue

        if args.merge and expected_save_names_per_dir is not None:
            # 检查该目录对应的所有预期结果文件是否都已存在
            all_exist = True
            for task_name, match_method in expected_save_names_per_dir.items():
                save_name = dir_name + '_' + match_method
                result_file = os.path.join(result_dir, save_name + '_metric_result.json')
                if not os.path.isfile(result_file):
                    all_exist = False
                    break
            if all_exist:
                print(f"跳过 {dir_name}: result 中已有完整结果")
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

    try:
        omni_md_path = write_omni_score_md(result_dir, saved_outputs_dir, args.config)
        print(f"已生成 Markdown 汇总表: {omni_md_path}")
    except Exception as e:
        print(f"生成 omni_scores.md 时出错（评测结果可能不完整）: {e}")

    # 将结果目录复制到 local/<saved_outputs 的目录名>，若已存在则合并（覆盖已有、保留未有）
    local_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'local')
    os.makedirs(local_base, exist_ok=True)
    output_dir_name = os.path.basename(os.path.normpath(saved_outputs_dir))
    target_dir = os.path.join(local_base, output_dir_name)
    shutil.copytree(result_dir, target_dir, dirs_exist_ok=True)
    print(f"结果已复制/合并到: {target_dir}")


if __name__ == '__main__':
    main()
