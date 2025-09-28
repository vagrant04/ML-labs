import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class PRMergePredictor:
    def __init__(self, data_path="input/merged_pr_data.xlsx"):
        """
        初始化PR合并预测器
        
        Args:
            data_path: 合并后的数据文件路径
        """
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """
        加载数据并进行基本预处理
        
        Returns:
            DataFrame: 加载的数据
        """
        try:
            df = pd.read_excel(self.data_path)
            print(f"成功加载数据，包含 {len(df)} 行，{len(df.columns)} 列")
            print(f"列名: {list(df.columns)}")
            return df
        except FileNotFoundError:
            print(f"错误: 找不到文件 {self.data_path}")
            return None
        except Exception as e:
            print(f"加载数据时出错: {str(e)}")
            return None
    
    def preprocess_data(self, df):
        """
        数据预处理
        
        Args:
            df: 原始数据框
            
        Returns:
            tuple: (特征矩阵X, 标签向量y, 特征名称)
        """
        # 检查是否有状态列（merged/closed）
        status_columns = ['status', 'state', 'merged', 'closed']
        target_column = None
        
        for col in status_columns:
            if col in df.columns:
                target_column = col
                break
        
        if target_column is None:
            print("警告: 未找到状态列，假设需要从其他信息推断")
            # 如果没有明确的状态列，可能需要从其他列推断
            # 这里假设有一个表示合并状态的列
            print("可用列:", list(df.columns))
            return None, None, None
        
        # 准备特征列
        feature_columns = [
            'directory_num', 'language_num', 'has_test', 'has_feature', 
            'has_bug', 'has_document', 'has_improve', 'has_refactor',
            'title_length', 'body_length', 'lines_added', 'lines_deleted',
            'segs_added', 'segs_deleted', 'segs_updated', 'files_added',
            'files_deleted', 'files_updated', 'commits', 'change_files'
        ]
        
        # 只选择存在的特征列
        available_features = [col for col in feature_columns if col in df.columns]
        print(f"使用的特征列: {available_features}")
        
        # 处理缺失值
        df_clean = df.copy()
        for col in available_features:
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].fillna('unknown')
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # 准备特征矩阵
        X = df_clean[available_features].copy()
        
        # 处理分类特征
        categorical_columns = X.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # 准备标签
        y = df_clean[target_column]
        if y.dtype == 'object':
            # 如果是字符串类型，进行编码
            y = self.label_encoder.fit_transform(y)
        
        print(f"特征矩阵形状: {X.shape}")
        print(f"标签分布: {pd.Series(y).value_counts()}")
        
        return X, y, available_features
    
    def time_based_split(self, df, X, y, test_size=0.2):
        """
        按时间顺序分割数据集
        
        Args:
            df: 原始数据框
            X: 特征矩阵
            y: 标签向量
            test_size: 测试集比例
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        # 寻找时间相关的列
        time_columns = ['created_at', 'updated_at', 'closed_at', 'merged_at', 'date', 'time']
        time_column = None
        
        for col in time_columns:
            if col in df.columns:
                time_column = col
                break
        
        if time_column is None:
            print("警告: 未找到时间列，使用随机分割")
            return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        
        # 按时间排序
        df_sorted = df.copy()
        try:
            df_sorted[time_column] = pd.to_datetime(df_sorted[time_column])
            df_sorted = df_sorted.sort_values(time_column)
            
            # 按时间顺序分割
            split_idx = int(len(df_sorted) * (1 - test_size))
            train_indices = df_sorted.index[:split_idx]
            test_indices = df_sorted.index[split_idx:]
            
            X_train = X.loc[train_indices]
            X_test = X.loc[test_indices]
            y_train = y[train_indices] if hasattr(y, 'loc') else y[train_indices]
            y_test = y[test_indices] if hasattr(y, 'loc') else y[test_indices]
            
            print(f"按时间分割 - 训练集: {len(X_train)} 样本, 测试集: {len(X_test)} 样本")
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            print(f"时间分割失败: {str(e)}，使用随机分割")
            return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    def train_models(self, X_train, y_train):
        """
        训练多个模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
        """
        # 标准化特征
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # 定义模型
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        print("开始训练模型...")
        for name, model in models.items():
            print(f"训练 {name}...")
            if name == 'RandomForest':
                # 随机森林不需要标准化
                model.fit(X_train, y_train)
            else:
                model.fit(X_train_scaled, y_train)
            self.models[name] = model
            print(f"{name} 训练完成")
    
    def evaluate_models(self, X_test, y_test):
        """
        评估模型性能
        
        Args:
            X_test: 测试特征
            y_test: 测试标签
        """
        X_test_scaled = self.scaler.transform(X_test)
        
        print("\n=== 模型评估结果 ===")
        for name, model in self.models.items():
            if name == 'RandomForest':
                y_pred = model.predict(X_test)
            else:
                y_pred = model.predict(X_test_scaled)
            
            # 计算F1分数（Macro平均）
            f1_macro = f1_score(y_test, y_pred, average='macro')
            f1_weighted = f1_score(y_test, y_pred, average='weighted')
            
            self.results[name] = {
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'predictions': y_pred
            }
            
            print(f"\n{name}:")
            print(f"  F1 Score (Macro): {f1_macro:.4f}")
            print(f"  F1 Score (Weighted): {f1_weighted:.4f}")
            print(f"  分类报告:")
            print(classification_report(y_test, y_pred, target_names=['Closed', 'Merged']))
    
    def plot_results(self):
        """
        绘制结果图表
        """
        if not self.results:
            print("没有结果可以绘制")
            return
        
        # F1分数比较
        models = list(self.results.keys())
        f1_scores = [self.results[model]['f1_macro'] for model in models]
        
        plt.figure(figsize=(10, 6))
        plt.bar(models, f1_scores)
        plt.title('模型F1分数比较 (Macro Average)')
        plt.ylabel('F1 Score')
        plt.ylim(0, 1)
        for i, score in enumerate(f1_scores):
            plt.text(i, score + 0.01, f'{score:.4f}', ha='center')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("结果图表已保存为 model_comparison.png")
    
    def run_experiment(self):
        """
        运行完整的实验流程
        """
        print("=== PR合并预测实验 ===")
        
        # 1. 加载数据
        df = self.load_data()
        if df is None:
            return
        
        # 2. 数据预处理
        X, y, feature_names = self.preprocess_data(df)
        if X is None:
            return
        
        # 3. 按时间分割数据
        X_train, X_test, y_train, y_test = self.time_based_split(df, X, y)
        
        # 4. 训练模型
        self.train_models(X_train, y_train)
        
        # 5. 评估模型
        self.evaluate_models(X_test, y_test)
        
        # 6. 绘制结果
        self.plot_results()
        
        # 7. 输出最佳模型
        best_model = max(self.results.keys(), key=lambda x: self.results[x]['f1_macro'])
        best_f1 = self.results[best_model]['f1_macro']
        print(f"\n最佳模型: {best_model} (F1 Macro: {best_f1:.4f})")

if __name__ == "__main__":
    # 运行实验
    predictor = PRMergePredictor()
    predictor.run_experiment()