# 特征提取模块
import pickle
import logging
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from src.utils.config import FileConfig, ModelConfig

logger = logging.getLogger(__name__)

class FeatureExtractor:
    def __init__(self, method='tfidf'):
        self.method = method
        self.vectorizer = None
        self._init_vectorizer()
    
    def _init_vectorizer(self):
        """初始化特征提取器"""
        if self.method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_df=ModelConfig.TFIDF_PARAMS['max_df'],
                min_df=ModelConfig.TFIDF_PARAMS['min_df']
            )
        elif self.method == 'count':
            self.vectorizer = CountVectorizer(
                max_df=ModelConfig.TFIDF_PARAMS['max_df'],
                min_df=ModelConfig.TFIDF_PARAMS['min_df']
            )
        else:
            raise ValueError("特征提取方法必须是 'tfidf' 或 'count'")
    
    def fit_transform(self, texts):
        """训练特征提取器并转换数据"""
        features = self.vectorizer.fit_transform(texts)
        logger.info(f"特征提取完成! 特征维度: {features.shape}")
        return features
    
    def transform(self, texts):
        """转换新数据"""
        return self.vectorizer.transform(texts)
    
    def save_vectorizer(self):
        """保存特征提取器"""
        with open(FileConfig.VECTORIZER_FILE, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        logger.info(f"特征提取器已保存至: {FileConfig.VECTORIZER_FILE}")
    
    def load_vectorizer(self):
        """加载特征提取器"""
        with open(FileConfig.VECTORIZER_FILE, 'rb') as f:
            self.vectorizer = pickle.load(f)
        logger.info("特征提取器加载成功!")