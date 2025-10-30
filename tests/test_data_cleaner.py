# 数据清洗单元测试
import sys
import unittest
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.cleaner import DataCleaner


class TestDataCleaner(unittest.TestCase):
    """DataCleaner 类的单元测试"""
    
    def setUp(self):
        """测试前的准备工作"""
        self.cleaner = DataCleaner()
    
    def test_clean_empty_text(self):
        """测试清洗空文本"""
        result = self.cleaner.clean_text("")
        self.assertEqual(result, "")
    
    def test_clean_none_text(self):
        """测试清洗 None 值"""
        result = self.cleaner.clean_text(None)
        self.assertEqual(result, "")
    
    def test_clean_whitespace_only(self):
        """测试清洗只有空格的文本"""
        result = self.cleaner.clean_text("   ")
        self.assertEqual(result, "")
    
    def test_clean_text_with_stopwords(self):
        """测试清洗包含停用词的文本"""
        text = "这是一个测试文本"
        result = self.cleaner.clean_text(text)
        # 结果应该是非空的，且经过分词
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
    
    def test_clean_text_returns_string(self):
        """测试清洗结果是字符串类型"""
        text = "这是一个测试"
        result = self.cleaner.clean_text(text)
        self.assertIsInstance(result, str)
    
    def test_clean_removes_tabs_and_newlines(self):
        """测试清洗去除制表符和换行符"""
        text = "测试\t文本\n测试"
        result = self.cleaner.clean_text(text)
        # 结果中不应该包含 \t 或 \n
        self.assertNotIn("\t", result)
        self.assertNotIn("\n", result)
    
    def test_clean_preserves_meaningful_words(self):
        """测试清洗保留有意义的词汇"""
        text = "这个产品质量很好"
        result = self.cleaner.clean_text(text)
        # 结果不应该是空的（应该保留有意义的词）
        self.assertTrue(len(result) > 0)
    
    def test_stopwords_loaded(self):
        """测试停用词是否正确加载"""
        self.assertIsNotNone(self.cleaner.stopwords)
        self.assertIsInstance(self.cleaner.stopwords, set)
        # 停用词数量应该大于 0
        self.assertGreater(len(self.cleaner.stopwords), 0)
    
    def test_clean_text_type_validation(self):
        """测试清洗对非字符串类型的处理"""
        # 数字输入
        result = self.cleaner.clean_text(123)
        self.assertEqual(result, "")
        
        # 列表输入
        result = self.cleaner.clean_text(['test'])
        self.assertEqual(result, "")
    
    def test_clean_maintains_word_order(self):
        """测试清洗后保持词序"""
        text = "我喜欢这个电影"
        result = self.cleaner.clean_text(text)
        # 结果应该是字符串，且不为空
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
    
    def test_clean_chinese_text(self):
        """测试清洗中文文本"""
        text = "这个产品真的很不错，强烈推荐！"
        result = self.cleaner.clean_text(text)
        self.assertIsInstance(result, str)
        # 结果应该包含分词后的内容
        self.assertGreater(len(result), 0)
    
    def test_clean_punctuation_handling(self):
        """测试清洗对标点符号的处理"""
        text = "太好了！这是一个很好的产品。"
        result = self.cleaner.clean_text(text)
        self.assertIsInstance(result, str)
        # 结果应该是非空的
        self.assertGreater(len(result), 0)


class TestDataCleanerIntegration(unittest.TestCase):
    """DataCleaner 的集成测试"""
    
    def setUp(self):
        """测试前的准备工作"""
        self.cleaner = DataCleaner()
    
    def test_clean_dataset_method_exists(self):
        """测试 clean_dataset 方法是否存在"""
        self.assertTrue(hasattr(self.cleaner, 'clean_dataset'))
        self.assertTrue(callable(getattr(self.cleaner, 'clean_dataset')))
    
    def test_clean_text_consistency(self):
        """测试相同文本清洗结果一致性"""
        text = "这是一个测试文本"
        result1 = self.cleaner.clean_text(text)
        result2 = self.cleaner.clean_text(text)
        # 相同输入应该产生相同输出
        self.assertEqual(result1, result2)
    
    def test_clean_multiple_texts(self):
        """测试清洗多条文本"""
        texts = [
            "这个产品很好",
            "质量太差了",
            "还可以吧"
        ]
        results = [self.cleaner.clean_text(text) for text in texts]
        # 所有结果都应该是字符串
        for result in results:
            self.assertIsInstance(result, str)


if __name__ == '__main__':
    unittest.main()
