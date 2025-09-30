# -*- coding: utf-8 -*-
import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
# 导入神经网络相关库
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tqdm import tqdm

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
warnings.filterwarnings('ignore')


class PRTTCAnalyzer:
    def __init__(self, data_dir='pr_data', result_dir='results'):
        """初始化PR关闭时长分析器，包含神经网络模型"""
        self.data_dir = data_dir
        self.result_dir = result_dir
        os.makedirs(self.result_dir, exist_ok=True)

        # 所有模型定义（新增神经网络模型）
        self.models = {
            '线性回归': LinearRegression(n_jobs=-1),
            '随机森林': RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            ),
            '梯度提升': GradientBoostingRegressor(
                n_estimators=100, max_depth=5, random_state=42
            ),
            'XGBoost': XGBRegressor(
                n_estimators=100, max_depth=5, random_state=42, n_jobs=-1
            ),
            'LightGBM': LGBMRegressor(
                n_estimators=100, max_depth=5, random_state=42, n_jobs=-1
            ),
            '神经网络(MLP)': self.build_mlp_model  # 神经网络模型（使用构建函数）
        }

        self.all_results = {}
        self.features = None
        self.data = None
        self.train_data = None
        self.test_data = None

    def build_mlp_model(self, input_dim):
        """构建多层感知器(MLP)神经网络模型"""
        model = Sequential([
            # 输入层和第一个隐藏层
            Dense(128, activation='relu', input_dim=input_dim),
            BatchNormalization(),
            Dropout(0.3),

            # 第二个隐藏层
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),

            # 第三个隐藏层
            Dense(32, activation='relu'),

            # 输出层（回归任务，无激活函数）
            Dense(1)
        ])

        # 编译模型
        model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['mean_absolute_error']
        )

        return model

    def load_and_merge_data(self, repo_names):
        """加载并合并多个仓库的数据"""
        dfs = []
        for repo in repo_names:
            owner, repo_name = repo.split('/')
            file_path = f"{self.data_dir}/{owner}_{repo_name}_ttc_features.csv"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                dfs.append(df)
                print(f"已加载 {repo} 数据，样本数: {len(df)}")
            else:
                print(f"警告: {file_path} 不存在，跳过该仓库")

        if not dfs:
            raise ValueError("没有找到任何有效数据文件，请先运行数据获取脚本")

        # 合并所有数据并按时间排序
        self.data = pd.concat(dfs, ignore_index=True)
        self.data['created_at'] = pd.to_datetime(self.data['created_at'])
        self.data = self.data.sort_values('created_at').reset_index(drop=True)

        print(f"合并后总样本数: {len(self.data)}")
        print(
            f"数据时间范围: {self.data['created_at'].min().strftime('%Y-%m-%d')} 至 {self.data['created_at'].max().strftime('%Y-%m-%d')}")
        return self.data

    def preprocess_data(self, test_size=0.2):
        """数据预处理与时间切分"""
        # 选择特征和目标变量
        exclude_cols = ['pr_number', 'created_at', 'repo', 'TTC']
        self.features = [col for col in self.data.columns if col not in exclude_cols]
        X = self.data[self.features]
        y = self.data['TTC']

        # 对目标变量进行对数转换（改善神经网络性能）
        self.y_log_transform = np.log1p(y)  # 使用log(1+x)避免0值问题

        # 时间序列切分（避免未来数据泄露）
        split_idx = int(len(self.data) * (1 - test_size))
        split_date = self.data.iloc[split_idx]['created_at']

        self.train_data = {
            'X': X.iloc[:split_idx],
            'y': y.iloc[:split_idx],
            'y_log': self.y_log_transform.iloc[:split_idx]  # 对数转换后的目标变量
        }
        self.test_data = {
            'X': X.iloc[split_idx:],
            'y': y.iloc[split_idx:],
            'y_log': self.y_log_transform.iloc[split_idx:]  # 对数转换后的目标变量
        }

        # 特征标准化（对神经网络尤为重要）
        self.scaler = StandardScaler()
        self.train_data['X_scaled'] = self.scaler.fit_transform(self.train_data['X'])
        self.test_data['X_scaled'] = self.scaler.transform(self.test_data['X'])

        print(f"使用的分割日期: {split_date.strftime('%Y-%m-%d')}")
        print(f"时间切分结果:")
        print(f"  训练集样本数: {len(self.train_data['X'])}")
        print(f"  测试集样本数: {len(self.test_data['X'])}")

        return self.train_data, self.test_data

    def evaluate_model(self, model_name, model, X_train, y_train, X_test, y_test, is_nn=False):
        """评估单个模型性能，支持神经网络特殊处理"""
        # 训练模型
        if is_nn:
            # 神经网络特殊训练流程
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )

            history = model.fit(
                X_train, y_train,
                epochs=100,
                batch_size=32,
                validation_split=0.1,
                callbacks=[early_stopping],
                verbose=0
            )
        else:
            # 传统机器学习模型训练
            model.fit(X_train, y_train)

        # 预测
        y_pred = model.predict(X_test)

        # 神经网络输出可能是二维数组，需要展平
        if is_nn:
            y_pred = y_pred.flatten()
            # 如果使用了对数转换，需要还原
            y_pred = np.expm1(y_pred)  # 与log1p对应，expm1(x) = exp(x) - 1
            y_true = np.expm1(y_test)  # 还原真实值用于评估
        else:
            y_true = y_test

        # 计算评估指标
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        # 保存结果
        result = {
            'model': model_name,
            'MAE': round(mae, 2),
            'MSE': round(mse, 2),
            'RMSE': round(rmse, 2),
            'R方': round(r2, 4),
            'y_true': y_true.tolist(),
            'y_pred': y_pred.tolist()
        }

        print(f"{model_name} 评估指标:")
        print(f"  MAE: {result['MAE']} 小时")
        print(f"  MSE: {result['MSE']}")
        print(f"  RMSE: {result['RMSE']} 小时")
        print(f"  R方: {result['R方']}\n")

        return result, model

    def compare_models(self):
        """比较所有模型的性能，包括神经网络"""
        print("\n============================================================")
        print("任务一：PR关闭时长（TTC）预测 - 多模型评估结果（含神经网络）")
        print("============================================================")

        # 所有模型结果
        results = {}

        # 线性回归使用标准化特征
        model_name = '线性回归'
        print(f"训练 {model_name}...")
        results[model_name], _ = self.evaluate_model(
            model_name,
            self.models[model_name],
            self.train_data['X_scaled'],
            self.train_data['y'],
            self.test_data['X_scaled'],
            self.test_data['y']
        )

        # 树模型使用原始特征
        for model_name in ['随机森林', '梯度提升', 'XGBoost', 'LightGBM']:
            print(f"训练 {model_name}...")
            results[model_name], _ = self.evaluate_model(
                model_name,
                self.models[model_name],
                self.train_data['X'],
                self.train_data['y'],
                self.test_data['X'],
                self.test_data['y']
            )

        # 神经网络模型（使用标准化特征和对数转换的目标变量）
        model_name = '神经网络(MLP)'
        print(f"训练 {model_name}...")
        # 构建适合输入维度的神经网络
        nn_model = self.models[model_name](input_dim=self.train_data['X_scaled'].shape[1])
        results[model_name], _ = self.evaluate_model(
            model_name,
            nn_model,
            self.train_data['X_scaled'],  # 神经网络需要标准化特征
            self.train_data['y_log'],  # 使用对数转换的目标变量
            self.test_data['X_scaled'],
            self.test_data['y_log'],
            is_nn=True  # 标记为神经网络，使用特殊处理
        )

        self.all_results['model_comparison'] = results
        return results

    def plot_model_comparison(self):
        """绘制包含神经网络的模型对比图"""
        results = self.all_results['model_comparison']
        metrics = ['MAE', 'RMSE', 'R方']
        data = []

        for model, stats in results.items():
            for metric in metrics:
                data.append({
                    '模型': model,
                    '指标': metric,
                    '值': stats[metric]
                })

        df = pd.DataFrame(data)

        plt.figure(figsize=(16, 6))
        sns.barplot(x='模型', y='值', hue='指标', data=df)
        plt.title('不同模型的性能指标对比（含神经网络）')
        plt.ylabel('值')
        plt.grid(alpha=0.3)
        plt.xticks(rotation=15)
        plt.tight_layout()

        save_path = f"{self.result_dir}/model_comparison.png"
        plt.savefig(save_path, dpi=300)
        print(f"模型对比图已保存: {save_path}")
        plt.close()

    def feature_ablation_experiment(self, top_n=10):
        """特征消融实验：逐步移除最重要的特征，观察模型性能变化"""
        print("\n============================================================")
        print("特征消融实验：逐步移除重要特征")
        print("============================================================")

        # 使用性能较好的XGBoost作为基准
        base_model = XGBRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
        base_model.fit(self.train_data['X'], self.train_data['y'])

        # 获取特征重要性并排序
        feature_importance = pd.DataFrame({
            '特征': self.features,
            '重要性': base_model.feature_importances_
        }).sort_values('重要性', ascending=False)

        # 保存特征重要性
        self.all_results['feature_importance'] = feature_importance.to_dict('records')
        print("原始特征重要性（前10名）:")
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
            print(f"  {i + 1}. {row['特征']}: {row['重要性']:.4f}")

        # 消融实验
        ablation_results = []
        remaining_features = feature_importance['特征'].tolist()

        # 基准模型（使用所有特征）
        _, base_model = self.evaluate_model(
            '基准模型（全部特征）',
            XGBRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1),
            self.train_data['X'],
            self.train_data['y'],
            self.test_data['X'],
            self.test_data['y']
        )
        ablation_results.append({
            '移除的特征': '无',
            '剩余特征数': len(remaining_features),
            'MAE': mean_absolute_error(self.test_data['y'], base_model.predict(self.test_data['X'])),
            'R方': r2_score(self.test_data['y'], base_model.predict(self.test_data['X']))
        })

        # 逐步移除最重要的特征
        for i in range(min(top_n, len(remaining_features))):
            removed_feature = remaining_features.pop(0)  # 移除最不重要的特征
            X_train_ablate = self.train_data['X'][remaining_features]
            X_test_ablate = self.test_data['X'][remaining_features]

            model = XGBRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
            model.fit(X_train_ablate, self.train_data['y'])
            y_pred = model.predict(X_test_ablate)

            mae = mean_absolute_error(self.test_data['y'], y_pred)
            r2 = r2_score(self.test_data['y'], y_pred)

            ablation_results.append({
                '移除的特征': removed_feature,
                '剩余特征数': len(remaining_features),
                'MAE': mae,
                'R方': r2
            })

            print(f"移除特征 '{removed_feature}' 后:")
            print(f"  剩余特征数: {len(remaining_features)}")
            print(f"  MAE: {mae:.2f} 小时")
            print(f"  R方: {r2:.4f}\n")

        self.all_results['ablation_experiment'] = ablation_results
        self.plot_ablation_results(ablation_results)
        return ablation_results

    def plot_ablation_results(self, ablation_results):
        """绘制特征消融实验结果"""
        df = pd.DataFrame(ablation_results)

        fig, ax1 = plt.subplots(figsize=(12, 6))

        # 绘制MAE
        color = 'tab:red'
        ax1.set_xlabel('实验步骤')
        ax1.set_ylabel('MAE (小时)', color=color)
        ax1.plot(df.index, df['MAE'], 'o-', color=color, label='MAE')
        ax1.tick_params(axis='y', labelcolor=color)

        # 创建第二个y轴绘制R方
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('R方', color=color)
        ax2.plot(df.index, df['R方'], 's-', color=color, label='R方')
        ax2.tick_params(axis='y', labelcolor=color)

        # 添加特征名称作为x轴标签
        plt.xticks(df.index, [f"步骤 {i}\n({r['移除的特征']})" for i, r in df.iterrows()], rotation=45)

        plt.title('特征消融实验结果')
        fig.tight_layout()

        save_path = f"{self.result_dir}/ablation_experiment.png"
        plt.savefig(save_path, dpi=300)
        print(f"特征消融实验图已保存: {save_path}")
        plt.close()

    def plot_prediction_comparison(self):
        """绘制预测值与真实值对比图（选择最佳模型）"""
        # 选择R方最高的模型
        best_model_name = max(
            self.all_results['model_comparison'].items(),
            key=lambda x: x[1]['R方']
        )[0]
        best_results = self.all_results['model_comparison'][best_model_name]

        # 绘制前100个样本的预测对比
        plt.figure(figsize=(12, 6))
        sample_size = min(100, len(best_results['y_true']))
        plt.plot(range(sample_size), best_results['y_true'][:sample_size], 'o-', label='真实值', alpha=0.7)
        plt.plot(range(sample_size), best_results['y_pred'][:sample_size], 's-', label='预测值', alpha=0.7)

        plt.title(f'最佳模型 ({best_model_name}) 预测值与真实值对比（前100样本）')
        plt.xlabel('样本索引')
        plt.ylabel('PR关闭时长（小时）')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        save_path = f"{self.result_dir}/prediction_comparison.png"
        plt.savefig(save_path, dpi=300)
        print(f"预测对比图已保存: {save_path}")
        plt.close()

    def save_results(self):
        """保存所有实验结果到JSON文件"""
        save_path = f"{self.result_dir}/experiment_results.json"
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.all_results, f, ensure_ascii=False, indent=2)
        print(f"所有实验结果已保存: {save_path}")

    def run_full_experiment(self, repo_names):
        """运行完整的拓展层实验流程，包含神经网络模型"""
        print("===== 开始拓展层（Level 3）实验（含神经网络） =====")

        # 1. 加载并合并多个仓库数据
        self.load_and_merge_data(repo_names)

        # 2. 数据预处理与切分
        self.preprocess_data()

        # 3. 多模型对比实验（含神经网络）
        self.compare_models()

        # 4. 特征消融实验
        self.feature_ablation_experiment()

        # 5. 绘制可视化图表
        self.plot_model_comparison()
        self.plot_prediction_comparison()

        # 6. 保存实验结果
        self.save_results()

        print("\n===== 拓展层（Level 3）实验完成 =====")


if __name__ == "__main__":
    # 配置参数
    REPOS = ["tensorflow/tensorflow", "pytorch/pytorch"]  # 至少两个项目的数据
    DATA_DIR = "pr_data"  # 数据存放目录
    RESULT_DIR = "level3_nn_results"  # 结果保存目录

    # 运行完整实验
    analyzer = PRTTCAnalyzer(data_dir=DATA_DIR, result_dir=RESULT_DIR)
    analyzer.run_full_experiment(REPOS)
