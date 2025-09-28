import pandas as pd
import numpy as np
from datetime import datetime
import re
import os

class PRDataProcessor:
    def __init__(self, input_path="input"):
        """
        初始化数据处理器
        
        Args:
            input_path: 输入文件夹路径，默认为"input"
        """
        self.input_path = input_path
        self.pr_features_file = os.path.join(input_path, "PR_features.xlsx")
        self.pr_info_file = os.path.join(input_path, "PR_info.xlsx")
        
    def load_pr_features(self):
        """
        从PR_features.xlsx中加载指定列
        
        Returns:
            DataFrame: 包含指定列的数据框
        """
        # 需要提取的列
        required_columns = [
            'number', 'directory_num', 'language_num', 'file_type', 
            'has_test', 'has_feature', 'has_bug', 'has_document', 
            'has_improve', 'has_refactor', 'title_length', 'body_length',
            'lines_added', 'lines_deleted', 'segs_added', 'segs_deleted',
            'segs_updated', 'files_added', 'files_deleted', 'files_updated'
        ]
        
        try:
            # 读取Excel文件
            df_features = pd.read_excel(self.pr_features_file)
            
            # 检查所需列是否存在
            missing_columns = [col for col in required_columns if col not in df_features.columns]
            if missing_columns:
                print(f"警告: PR_features.xlsx中缺少以下列: {missing_columns}")
                # 只选择存在的列
                available_columns = [col for col in required_columns if col in df_features.columns]
                df_features = df_features[available_columns]
            else:
                df_features = df_features[required_columns]
                
            print(f"成功加载PR_features.xlsx，包含 {len(df_features)} 行数据")
            return df_features
            
        except FileNotFoundError:
            print(f"错误: 找不到文件 {self.pr_features_file}")
            return None
        except Exception as e:
            print(f"读取PR_features.xlsx时出错: {str(e)}")
            return None
    
    def load_pr_info(self):
        """
        从PR_info.xlsx中加载指定列
        
        Returns:
            DataFrame: 包含commits和change_files列的数据框
        """
        required_columns = ['created_at', 'commits', 'changed_files', 'merged']
        
        try:
            # 读取Excel文件
            df_info = pd.read_excel(self.pr_info_file)
            
            # 检查所需列是否存在
            missing_columns = [col for col in required_columns if col not in df_info.columns]
            if missing_columns:
                print(f"警告: PR_info.xlsx中缺少以下列: {missing_columns}")
                # 只选择存在的列
                available_columns = [col for col in required_columns if col in df_info.columns]
                df_info = df_info[available_columns]
            else:
                df_info = df_info[required_columns]
                
            print(f"成功加载PR_info.xlsx，包含 {len(df_info)} 行数据")
            return df_info
            
        except FileNotFoundError:
            print(f"错误: 找不到文件 {self.pr_info_file}")
            return None
        except Exception as e:
            print(f"读取PR_info.xlsx时出错: {str(e)}")
            return None
    
    def merge_data(self):
        """
        合并两个数据框
        
        Returns:
            DataFrame: 合并后的数据框
        """
        # 加载两个数据文件
        df_features = self.load_pr_features()
        df_info = self.load_pr_info()
        
        if df_features is None or df_info is None:
            print("错误: 无法加载数据文件")
            return None
        
        # 检查数据行数是否一致
        if len(df_features) != len(df_info):
            print(f"警告: 两个文件的行数不一致 - PR_features: {len(df_features)}, PR_info: {len(df_info)}")
            print("将按行索引进行拼接")
        
        # 按列拼接数据（假设两个文件的行顺序对应）
        merged_df = pd.concat([df_features, df_info], axis=1)
        
        print(f"数据合并完成，最终数据包含 {len(merged_df)} 行，{len(merged_df.columns)} 列")
        print(f"列名: {list(merged_df.columns)}")
        
        return merged_df
    
    def save_merged_data(self, output_file=os.path.join("input", "merged_pr_data.xlsx")):
        """
        保存合并后的数据
        
        Args:
            output_file: 输出文件名，默认为"merged_pr_data.xlsx"
        """
        merged_df = self.merge_data()
        
        if merged_df is not None:
            try:
                merged_df.to_excel(output_file, index=False)
                print(f"合并后的数据已保存到: {output_file}")
                return True
            except Exception as e:
                print(f"保存文件时出错: {str(e)}")
                return False
        else:
            print("无法保存数据，合并失败")
            return False

if __name__ == "__main__":
    # 示例使用
    processor = PRDataProcessor()
    
    # 合并数据并保存
    merged_data = processor.merge_data()
    
    if merged_data is not None:
        # 显示数据基本信息
        print("\n=== 数据概览 ===")
        print(f"数据形状: {merged_data.shape}")
        print(f"列名: {list(merged_data.columns)}")
        print("\n前5行数据:")
        print(merged_data.head())
        
        # 保存到文件
        processor.save_merged_data(os.path.join("input", "merged_pr_data.xlsx"))
    else:
        print("数据处理失败")
    