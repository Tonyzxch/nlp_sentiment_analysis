# 数据预处理模块
import pandas as pd
import logging
from src.utils.config import FileConfig

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        pass
    
    def create_dataset(self, cleaned_comments):
        """创建训练数据集"""
        logger.info("开始创建数据集...")
        
        try:
            # 读取标签数据
            with open(FileConfig.RAW_LABELS_FILE, 'r', encoding='utf-8') as f:
                label_lines = f.readlines()
            
            # 处理标签（假设标签是逗号分隔的）
            all_labels = []
            for line in label_lines:
                labels = line.strip().split(',')
                all_labels.extend([label.strip() for label in labels if label.strip()])
            
            # 确保数据对齐
            min_length = min(len(cleaned_comments), len(all_labels))
            aligned_comments = cleaned_comments[:min_length]
            aligned_labels = all_labels[:min_length]
            
            # 创建DataFrame
            df = pd.DataFrame({
                'text': aligned_comments,
                'label': aligned_labels
            })
            
            # 转换标签为整数类型
            try:
                df['label'] = df['label'].astype(int)
            except ValueError:
                logger.warning("标签转换失败，保持原类型")
            
            # 保存数据集
            df.to_csv(FileConfig.PROCESSED_CSV_FILE, index=False)
            logger.info(f"数据集创建完成! 共 {len(df)} 条数据")
            logger.info(f"标签分布:\n{df['label'].value_counts()}")
            
            return df
            
        except Exception as e:
            logger.error(f"数据集创建过程中发生错误: {str(e)}")
            raise