# 进行新文本预测​
import pickle
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.cleaner import DataCleaner
from src.utils.config import FileConfig

class SentimentPredictor:
    def __init__(self):
        self.cleaner = DataCleaner()
        self.load_models()
    
    def load_models(self):
        """加载训练好的模型和特征提取器"""
        with open(FileConfig.VECTORIZER_FILE, 'rb') as f:
            self.vectorizer = pickle.load(f)
        with open(FileConfig.TRAINED_MODEL_FILE, 'rb') as f:
            self.model = pickle.load(f)
    
    def predict(self, text):
        """预测单条文本的情感"""
        # 清洗文本
        cleaned_text = self.cleaner.clean_text(text)
        # 提取特征
        features = self.vectorizer.transform([cleaned_text])
        # 预测
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0]
        
        # 确保预测值是整数进行比较
        pred_int = int(prediction) if isinstance(prediction, (str, float)) else prediction
        sentiment = "Positive" if pred_int == 1 else "Negative"
        confidence = probability[1] if pred_int == 1 else probability[0]
        
        return {
            'text': text,
            'cleaned_text': cleaned_text,
            'sentiment': sentiment,
            'confidence': round(confidence * 100, 2),
            'raw_prediction': pred_int
        }

# 使用示例
if __name__ == "__main__":
    predictor = SentimentPredictor()
    
    test_texts = [
        "这个电影真是太精彩了，演员演技很棒！",
        "产品质量很差，用了一次就坏了",
        "还行吧，没什么特别的感觉"
    ]
    
    print("=== 情感预测演示 ===")
    for text in test_texts:
        result = predictor.predict(text)
        print(f"原文: {result['text']}")
        print(f"情感: {result['sentiment']} (置信度: {result['confidence']}%)")
        print(f"清洗后: {result['cleaned_text']}")
        print("-" * 50)