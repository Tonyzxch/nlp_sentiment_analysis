# 项目配置文件
import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 数据路径
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# 输出路径
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = OUTPUT_DIR / "models"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
LOGS_DIR = OUTPUT_DIR / "logs"

# 创建必要的目录
for directory in [PROCESSED_DATA_DIR, MODELS_DIR, PREDICTIONS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# 文件路径配置
class FileConfig:
    # 输入文件
    STOPWORDS_FILE = RAW_DATA_DIR / "HGD_StopWords.txt"
    RAW_COMMENTS_FILE = RAW_DATA_DIR / "ALL_Comment.txt"
    RAW_LABELS_FILE = RAW_DATA_DIR / "All_label.txt"
    
    # 输出文件
    CLEANED_COMMENTS_FILE = PROCESSED_DATA_DIR / "cleaned_comments.txt"
    PROCESSED_CSV_FILE = PROCESSED_DATA_DIR / "sentiment_dataset.csv"
    
    # 模型文件
    VECTORIZER_FILE = MODELS_DIR / "tfidf_vectorizer.pkl"
    TRAINED_MODEL_FILE = MODELS_DIR / "sentiment_model.pkl"

# 模型参数配置
class ModelConfig:
    TEST_SIZE = 0.1
    RANDOM_STATE = 42
    TFIDF_PARAMS = {
        'max_df': 0.8,
        'min_df': 3
    }
    LOGISTIC_REGRESSION_PARAMS = {
        'max_iter': 1000,
        'C': 1.0
    }