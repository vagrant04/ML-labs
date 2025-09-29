import sys
sys.path.append('src')

from data_processing import PRDataProcessor
from train import PRMergePredictor

TOKEN = "your_github_token"

def main():
    # 数据处理
    processor = PRDataProcessor(input_path="input")
    processor.save_merged_data("input/merged_pr_data.xlsx")
    
    # 模型训练
    predictor = PRMergePredictor(data_path="input/merged_pr_data.xlsx")
    predictor.run_experiment()

if __name__ == "__main__":
    main()