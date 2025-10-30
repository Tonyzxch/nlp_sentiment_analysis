# 模型训练模块
import pickle
import logging
from sklearn import linear_model, model_selection
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import pandas as pd
from src.utils.config import FileConfig, ModelConfig

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.model = None
        self.best_params = None
    
    def prepare_data(self, df):
        """准备训练数据"""
        X = df['text'].values.astype('U')
        y = df['label'].values
        
        # 划分训练测试集
        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            X, y, 
            test_size=ModelConfig.TEST_SIZE,
            random_state=ModelConfig.RANDOM_STATE,
            stratify=y  # 保持标签分布
        )
        
        logger.info(f"数据划分完成: 训练集 {X_train.shape}, 测试集 {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train, use_grid_search=True):
        """训练模型"""
        logger.info("开始训练模型...")
        
        # 初始化模型
        base_model = linear_model.LogisticRegression(
            max_iter=ModelConfig.LOGISTIC_REGRESSION_PARAMS['max_iter']
        )
        
        if use_grid_search:
            # 使用网格搜索寻找最优参数
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2']
            }
            
            self.model = GridSearchCV(
                base_model, param_grid, 
                cv=3, scoring='accuracy', n_jobs=-1
            )
        else:
            self.model = base_model
        
        # 训练模型
        self.model.fit(X_train, y_train)
        
        if use_grid_search:
            self.best_params = self.model.best_params_
            logger.info(f"最优参数: {self.best_params}")
        
        logger.info("模型训练完成!")
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """评估模型性能"""
        y_pred = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        logger.info(f"测试集准确率: {accuracy:.4f}")
        logger.info("分类报告:\n" + report)
        
        return accuracy, report
    
    def save_model(self):
        """保存训练好的模型"""
        with open(FileConfig.TRAINED_MODEL_FILE, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"模型已保存至: {FileConfig.TRAINED_MODEL_FILE}")
    
    def load_model(self):
        """加载模型"""
        with open(FileConfig.TRAINED_MODEL_FILE, 'rb') as f:
            self.model = pickle.load(f)
        logger.info("模型加载成功!")