import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse

class CSVWeightedVotingEnsemble:
    def __init__(self, model_scores):
        """
        基于CSV文件的加权投票集成
        
        Args:
            model_scores: 字典，包含模型名称和对应的Kaggle得分
        """
        self.model_scores = model_scores
        self.normalize_weights()
        
    def normalize_weights(self):
        """归一化模型权重"""
        total_score = sum(self.model_scores.values())
        self.model_weights = {model: score/total_score for model, score in self.model_scores.items()}
        print("模型权重归一化结果:")
        for model, weight in self.model_weights.items():
            print(f"  {model}: {weight:.4f}")
    
    def load_predictions(self, csv_directory="./submissions"):
        """
        从CSV文件加载所有模型的预测结果
        
        Args:
            csv_directory: CSV文件所在目录
        """
        self.predictions = {}
        self.sample_count = None
        
        for model_name in self.model_scores.keys():
            csv_path = Path(csv_directory) / f"{model_name}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                # 确保有index和pred列
                if 'index' in df.columns and 'pred' in df.columns:
                    self.predictions[model_name] = df['pred'].values
                    if self.sample_count is None:
                        self.sample_count = len(df)
                    elif len(df) != self.sample_count:
                        raise ValueError(f"模型 {model_name} 的样本数量不一致")
                else:
                    raise ValueError(f"CSV文件 {csv_path} 缺少必要的列")
            else:
                print(f"警告: 未找到文件 {csv_path}")
        
        print(f"成功加载 {len(self.predictions)} 个模型的预测结果，共 {self.sample_count} 个样本")
    
    def weighted_voting(self):
        """执行加权投票集成"""
        if not hasattr(self, 'predictions'):
            raise ValueError("请先加载预测结果")
        
        # 确定类别数量（从所有预测中找出最大类别值）
        max_class = max([np.max(preds) for preds in self.predictions.values()])
        num_classes = int(max_class) + 1
        print(f"检测到 {num_classes} 个类别")
        
        # 初始化最终预测结果
        final_predictions = np.zeros(self.sample_count, dtype=int)
        
        # 对每个样本进行加权投票
        for i in range(self.sample_count):
            class_weights = np.zeros(num_classes)
            
            # 收集每个模型对该样本的投票
            for model_name, preds in self.predictions.items():
                if model_name in self.model_weights:
                    predicted_class = preds[i]
                    weight = self.model_weights[model_name]
                    class_weights[predicted_class] += weight
            
            # 选择权重最高的类别
            final_predictions[i] = np.argmax(class_weights)
        
        return final_predictions
    
    def create_ensemble_submission(self, output_file="weighted_ensemble_submission.csv"):
        """生成集成提交文件"""
        final_predictions = self.weighted_voting()
        
        submission_df = pd.DataFrame({
            'index': range(len(final_predictions)),
            'pred': final_predictions
        })
        
        submission_df.to_csv(output_file, index=False)
        print(f"集成提交文件已保存: {output_file}")
        return submission_df
    
    def analyze_ensemble(self):
        """分析集成结果"""
        if not hasattr(self, 'predictions'):
            raise ValueError("请先加载预测结果")
        
        print("\n=== 集成分析 ===")
        
        # 计算模型一致性
        sample_agreements = []
        for i in range(self.sample_count):
            sample_preds = [preds[i] for preds in self.predictions.values()]
            unique_preds = len(set(sample_preds))
            sample_agreements.append(unique_preds)
        
        full_agreement = sum(1 for x in sample_agreements if x == 1) / self.sample_count
        majority_disagreement = sum(1 for x in sample_agreements if x > 1) / self.sample_count
        
        print(f"完全一致样本比例: {full_agreement:.4f}")
        print(f"存在分歧样本比例: {majority_disagreement:.4f}")
        
        # 显示前几个样本的投票情况
        print("\n前5个样本的投票情况:")
        for i in range(min(5, self.sample_count)):
            sample_votes = {}
            for model_name, preds in self.predictions.items():
                vote = preds[i]
                if vote not in sample_votes:
                    sample_votes[vote] = []
                sample_votes[vote].append(model_name)
            
            print(f"样本 {i}:")
            for cls, models in sample_votes.items():
                total_weight = sum(self.model_weights[model] for model in models)
                print(f"  类别 {cls}: {len(models)} 个模型, 总权重 {total_weight:.4f}")

def main():
    # 您的模型得分配置
    model_scores = {
        'densenet121_submission': 0.90245,
        'densenet201_submission': 0.8777, 
        'swin_submission': 0.76962,
        'densenet201_submission1': 0.73137,
        'densenet161_submission': 0.71426,
        'densenet161_submission1': 0.70678,
    }
    
    # 创建集成器
    ensemble = CSVWeightedVotingEnsemble(model_scores)
    
    # 加载预测结果（假设CSV文件在当前目录的submissions文件夹中）
    try:
        ensemble.load_predictions("./sub")
    except Exception as e:
        print(f"加载预测结果时出错: {e}")
        print("请确保CSV文件存在于 ./submissions/ 目录下")
        return
    
    # 生成集成提交文件
    submission_df = ensemble.create_ensemble_submission("final_ensemble_submission.csv")
    
    # 分析集成结果
    ensemble.analyze_ensemble()
    
    print("\n集成完成！请将 final_ensemble_submission.csv 提交到Kaggle")

if __name__ == "__main__":
    main()
