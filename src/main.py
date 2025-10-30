# 主程序入口
import os
import sys
import logging
from pathlib import Path

# 将项目根目录添加到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.cleaner import DataCleaner
from src.data.processor import DataProcessor
from src.features.extractor import FeatureExtractor
from src.models.trainer import ModelTrainer
from src.utils.config import FileConfig, LOGS_DIR

def setup_logging():
    """配置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOGS_DIR / 'sentiment_analysis.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    """主函数"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("=== 情感分析项目开始执行 ===")
        
        # 1. 数据清洗
        cleaner = DataCleaner()
        cleaned_comments = cleaner.clean_dataset()
        
        # 2. 数据预处理
        processor = DataProcessor()
        df = processor.create_dataset(cleaned_comments)
        
        # 3. 准备数据
        trainer = ModelTrainer()
        X_train, X_test, y_train, y_test = trainer.prepare_data(df)
        
        # 4. 特征提取
        feature_extractor = FeatureExtractor(method='tfidf')
        X_train_features = feature_extractor.fit_transform(X_train)
        X_test_features = feature_extractor.transform(X_test)
        
        # 5. 模型训练
        model = trainer.train_model(X_train_features, y_train, use_grid_search=True)
        
        # 6. 模型评估
        accuracy, report = trainer.evaluate_model(X_test_features, y_test)
        
        # 7. 保存模型和特征提取器
        trainer.save_model()
        feature_extractor.save_vectorizer()
        
        logger.info("=== 项目执行完成 ===")
        
    except Exception as e:
        logger.error(f"项目执行过程中发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    main()