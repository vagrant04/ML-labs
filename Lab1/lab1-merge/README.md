# GitHub PR合并预测实验

基于GitHub API数据的Pull Request合并预测机器学习实验。

## 环境要求

```bash
pip install pandas numpy scikit-learn matplotlib seaborn requests openpyxl
```
## 快速开始
``` bash
# 直接运行主函数即可
python main.py
```

## 设计思路

### 1. 数据获取

如果需要重新获取数据：

```bash
# 修改 src/data_processing.py 中的 GitHub token
# 然后运行数据获取
python src/data_processing.py
```

**注意**：GitHub API有速率限制，建议使用提供的数据文件。

### 2. 数据合并

```python
from src.data_processing import GitHubPRAnalyzerConcurrent

# 合并多个数据源
GitHubPRAnalyzerConcurrent.merge_excel_files(
    "input/yii2.xlsx", 
    "input/pytorch.xlsx", 
    "input/merged_pr_data.xlsx"
)
```

### 3. 模型训练与评估

```bash
python src/train.py
```

## 文件结构

```
├── src/
│   ├── data_processing.py    # 数据获取与处理
│   └── train.py             # 模型训练与评估
├── input/                   # 数据文件目录
├── main.py                  # 主函数
└── README.md
```

## 输出结果

运行后生成：
- `model_metrics_comparison.png` - 性能对比图
- `model_radar_chart.png` - 雷达图
- `confusion_matrices.png` - 混淆矩阵
- `model_evaluation_results.xlsx` - 详细结果


## 注意事项

1. GitHub token需要有repo访问权限
2. 数据获取可能需要较长时间（API限制）