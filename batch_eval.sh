#!/bin/bash
# 批量评估脚本
# 遍历 saved_outputs 下的所有目录，找到 *_md 子目录进行评估

SAVED_OUTPUTS_DIR="/export/home/tangzihan.15/fast-ocr/local/saved_outputs"
CONFIG_TEMPLATE="/export/home/tangzihan.15/OmniDocBench/configs/end2end.yaml"
WORK_DIR="/export/home/tangzihan.15/OmniDocBench"

cd "$WORK_DIR"

# 遍历所有目录
for dir in "$SAVED_OUTPUTS_DIR"/*/; do
    # 跳过非目录
    [ -d "$dir" ] || continue
    
    dir_name=$(basename "$dir")
    
    # 查找 *_md 子目录
    md_dir=$(find "$dir" -maxdepth 1 -type d -name "*_md" | head -1)
    
    if [ -z "$md_dir" ]; then
        echo "跳过 $dir_name: 未找到 *_md 子目录"
        continue
    fi
    
    echo "=========================================="
    echo "正在评估: $dir_name"
    echo "结果目录: $md_dir"
    echo "=========================================="
    
    # 创建临时配置文件
    temp_config="/tmp/end2end_${dir_name}.yaml"
    
    # 使用 sed 替换 data_path
    sed "s|data_path: ./demo_data/end2end|data_path: $md_dir|g" "$CONFIG_TEMPLATE" > "$temp_config"
    
    # 运行评估
    python pdf_validation.py --config "$temp_config"
    
    # 清理临时配置
    rm -f "$temp_config"
    
    echo ""
done

echo "批量评估完成！"
echo "结果保存在 $WORK_DIR/result/ 目录下"
