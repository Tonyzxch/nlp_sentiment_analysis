# 数据清洗模块
import jieba
import time
import logging
from pathlib import Path
from src.utils.config import FileConfig

logger = logging.getLogger(__name__)

class DataCleaner:
    def __init__(self):
        self.stopwords = self._load_stopwords()
    
    def _load_stopwords(self):
        """加载停用词表"""
        try:
            with open(FileConfig.STOPWORDS_FILE, 'r', encoding='UTF-8') as f:
                stopwords = [line.strip() for line in f if line.strip()]
            logger.info(f"成功加载 {len(stopwords)} 个停用词")
            return set(stopwords)
        except FileNotFoundError:
            logger.error(f"停用词文件不存在: {FileConfig.STOPWORDS_FILE}")
            return set()
    
    def clean_text(self, text):
        """清洗单条文本"""
        if not text or not isinstance(text, str):
            return ""
        
        # 分词
        words = jieba.cut(text.strip())
        # 去停用词和空白字符
        cleaned_words = [
            word for word in words 
            if word.strip() and word not in self.stopwords and word != '\t'
        ]
        return ' '.join(cleaned_words)
    
    def clean_dataset(self):
        """清洗整个数据集"""
        logger.info("开始数据清洗...")
        start_time = time.time()
        
        try:
            # 读取原始评论
            with open(FileConfig.RAW_COMMENTS_FILE, 'r', encoding='UTF-8') as f:
                raw_comments = f.readlines()
            
            # 清洗数据
            cleaned_comments = []
            for i, comment in enumerate(raw_comments):
                cleaned_comment = self.clean_text(comment)
                if cleaned_comment:  # 只保留非空结果
                    cleaned_comments.append(cleaned_comment)
                
                if (i + 1) % 1000 == 0:
                    logger.info(f"已处理 {i + 1} 条数据")
            
            # 保存清洗后的数据
            with open(FileConfig.CLEANED_COMMENTS_FILE, 'w', encoding='UTF-8') as f:
                for comment in cleaned_comments:
                    f.write(comment + '\n')
            
            processing_time = time.time() - start_time
            logger.info(f"数据清洗完成! 共处理 {len(cleaned_comments)} 条数据, 耗时 {processing_time:.2f} 秒")
            return cleaned_comments
            
        except Exception as e:
            logger.error(f"数据清洗过程中发生错误: {str(e)}")
            raise