# NLP 情感分析项目

## 📋 项目概述

这是一个基于**机器学习**的中文情感分析系统，能够对中文文本进行情感分类（正面/负面）。项目使用 **TF-IDF** 特征提取和 **逻辑回归** 模型进行分类。

**主要特性：**
- ✅ 中文文本数据清洗和预处理
- ✅ TF-IDF 特征提取和 Count 向量化
- ✅ 逻辑回归模型训练和优化
- ✅ 多模型对比实验 (6种算法)
- ✅ 网格搜索超参数调优
- ✅ 模型评估和可视化分析
- ✅ 单元测试覆盖
- ✅ 完整的预测和推理接口

---

## 📁 项目结构

```
nlp_sentiment_analysis/
│
├── data/                          # 数据目录
│   ├── raw/                       # 原始数据
│   │   ├── HGD_StopWords.txt      # 停用词表
│   │   ├── ALL_Comment.txt        # 原始评论数据
│   │   └── All_label.txt          # 评论标签
│   └── processed/                 # 处理后的数据
│       ├── cleaned_comments.txt   # 清洗后的评论
│       └── sentiment_dataset.csv  # 处理后的数据集
│
├── src/                           # 源代码目录
│   ├── data/                      # 数据处理模块
│   │   ├── __init__.py
│   │   ├── cleaner.py             # 数据清洗
│   │   └── processor.py           # 数据预处理
│   ├── features/                  # 特征工程模块
│   │   ├── __init__.py
│   │   └── extractor.py           # 特征提取
│   ├── models/                    # 模型模块
│   │   ├── __init__.py
│   │   └── trainer.py             # 模型训练和评估
│   ├── utils/                     # 工具模块
│   │   ├── __init__.py
│   │   └── config.py              # 配置管理
│   ├── main.py                    # 主程序入口
│   ├── analysis.py                # 结果分析和可视化
│   └── predict.py                 # 预测推理接口
│
├── notebooks/                     # Jupyter 笔记本
│   └── exploration.ipynb          # 数据探索分析
│
├── tests/                         # 单元测试
│   ├── __init__.py
│   └── test_data_cleaner.py       # 数据清洗测试
│
├── output/                            # 输出结果目录
│   ├── models/                        # 训练好的模型
│   │   ├── sentiment_model.pkl        # 逻辑回归模型
│   │   └── tfidf_vectorizer.pkl       # TF-IDF 向量化器
│   ├── logs/                          # 日志文件
│   │   └── sentiment_analysis.log     # 模型训练日志
│   ├── predictions/                   # 预测结果 (预留目录)
│   ├── confusion_matrix.png           # 最佳模型的混淆矩阵
│   ├── model_comparison.png           # 多模型对比图表
│   ├── model_comparison_results.csv   # 模型对比结果 (CSV)
│   └── results_analysis.png           # 数据分析结果图表
│
├── requirements.txt               # Python 依赖
├── setup.py                       # 安装脚本
└── README.md                      # 项目说明文档
```

---

## 🚀 快速开始

### 1. 环境配置

**Python 版本:** >= 3.8

**创建虚拟环境 (可选但推荐):**
```bash
# Windows
python -m venv nlp_env
nlp_env\Scripts\activate

# Linux/Mac
python -m venv nlp_env
source nlp_env/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

**依赖包：**
- `jieba>=0.42.1` - 中文分词
- `pandas>=1.5.0` - 数据处理
- `scikit-learn>=1.2.0` - 机器学习库
- `numpy>=1.24.0` - 数值计算
- `matplotlib` - 数据可视化
- `seaborn` - 统计可视化

### 3. 准备数据

将您的数据文件放在 `data/raw/` 目录下：
- `ALL_Comment.txt` - 评论文本（每行一条）
- `All_label.txt` - 对应的标签（逗号分隔）
- `HGD_StopWords.txt` - 停用词表（每行一个）

### 4. 运行训练

```bash
cd src
python main.py
```

**执行步骤：**
1. 数据清洗 - 分词、去停用词
2. 数据预处理 - 创建数据集
3. 数据划分 - 训练/测试集分割
4. 特征提取 - TF-IDF 向量化
5. 模型训练 - 逻辑回归 + 网格搜索
6. 模型评估 - 准确率和分类报告
7. 保存模型 - 持久化训练结果

### 5. 结果分析

```bash
python analysis.py
```

生成可视化报告：
- 标签分布图
- 文本长度分布图
- 模型预测结果图

---

## 📊 核心模块说明

### **数据清洗 (data/cleaner.py)**

```python
from src.data.cleaner import DataCleaner

cleaner = DataCleaner()
cleaned_text = cleaner.clean_text("这是一条评论")
```

**功能：**
- 中文分词（使用 jieba）
- 停用词过滤
- 特殊字符处理

### **特征提取 (features/extractor.py)**

```python
from src.features.extractor import FeatureExtractor

extractor = FeatureExtractor(method='tfidf')
features = extractor.fit_transform(texts)
```

**特征方法：**
- TF-IDF (推荐)
- Count Vectorizer

### **模型训练 (models/trainer.py)**

```python
from src.models.trainer import ModelTrainer

trainer = ModelTrainer()
X_train, X_test, y_train, y_test = trainer.prepare_data(df)
model = trainer.train_model(X_train, y_train, use_grid_search=True)
accuracy, report = trainer.evaluate_model(X_test, y_test)
```

**模型：** 逻辑回归 (Logistic Regression)

### **预测推理 (predict.py)**

```python
from src.predict import SentimentPredictor

predictor = SentimentPredictor()
result = predictor.predict("这个产品很不错！")
# 输出: {'text': '...', 'sentiment': 'Positive', 'confidence': 92.5, ...}
```

---

## 🎯 核心模块速查表

| 模块 | 文件 | 功能 | 主要函数 |
|------|------|------|--------|
| **数据清洗** | `src/data/cleaner.py` | 分词 + 去停用词 | `clean_text()` |
| **数据处理** | `src/data/processor.py` | 创建数据集 | `create_dataset()` |
| **特征提取** | `src/features/extractor.py` | 向量化 | `fit_transform()` |
| **模型训练** | `src/models/trainer.py` | 模型 + 评估 | `train_model()` |
| **预测接口** | `src/predict.py` | 单文本预测 | `predict()` |
| **分析可视化** | `src/analysis.py` | 图表生成 | `plot_results()` |
| **配置管理** | `src/utils/config.py` | 参数配置 | `FileConfig`, `ModelConfig` |

---

## 💾 重要文件位置

| 用途 | 文件路径 |
|------|--------|
| **项目说明** | `README.md` |
| **完整总结** | `PROJECT_SUMMARY.md` |
| **依赖清单** | `requirements.txt` |
| **训练后模型** | `output/models/sentiment_model.pkl` |
| **TF-IDF 向量器** | `output/models/tfidf_vectorizer.pkl` |
| **模型对比结果** | `output/model_comparison_results.csv` |
| **混淆矩阵图** | `output/confusion_matrix.png` |
| **性能对比图** | `output/model_comparison.png` |
| **分析结果图** | `output/results_analysis.png` |
| **训练日志** | `output/logs/sentiment_analysis.log` |

---

## 🧪 单元测试

### 运行测试

```bash
# 在 tests 目录下
cd tests
python test_data_cleaner.py

# 或使用 unittest
python -m unittest test_data_cleaner -v

# 或使用 pytest
pytest test_data_cleaner.py -v
```

### 测试覆盖

- ✅ 数据清洗功能测试 (11 个用例)
- ✅ 集成测试 (3 个用例)
- ✅ 边界值测试
- ✅ 类型验证测试

---

## 📈 模型性能

### 单一模型性能 (逻辑回归)

| 指标 | 数值 |
|------|------|
| 训练集大小 | 7831 条 |
| 正面样本 | 5070 条 (64.8%) |
| 负面样本 | 2761 条 (35.2%) |
| 测试集准确率 | ~95.66% |
| 特征提取方法 | TF-IDF 向量 |
| 模型类型 | 逻辑回归 |

### 多模型对比结果

对比了 6 种机器学习模型的性能，详见 `output/model_comparison_results.csv`：

| 模型 | 特征方法 | 测试准确率 | 训练时间 | 过拟合 | 备注 |
|------|---------|----------|--------|------|------|
| Logistic Regression | TF-IDF | 96.30% | 0.046s | 低 | ⭐ 最佳选择 |
| Random Forest | Count | 93.37% | 0.904s | 中等 | 准确率高但易过拟合 |
| Naive Bayes | Count | 92.09% | 0.012s | 低 | 最快但准确率不如LR |
| Decision Tree | TF-IDF | 91.96% | 0.569s | 中等 | 过拟合较严重 |
| SVM | Count | 65.05% | 5.858s | 低 | ❌ 效果不佳，耗时最长 |
| KNN | TF-IDF | 43.11% | 1.888s | 低 | ❌ 效果很差 |

**关键发现：**
- ✅ **逻辑回归** 是最优选择：最高准确率 (96.30%) + 快速训练 (0.046s)
- ⚠️ **KNN 和 SVM** 在这个任务上表现不佳
- 📊 稀疏向量特征 (TF-IDF/Count) 对 KNN 性能影响较大
- 🎯 **结论**：建议生产环境使用逻辑回归

---

## 🔧 配置参数

编辑 `src/utils/config.py` 修改参数：

```python
class ModelConfig:
    TEST_SIZE = 0.1              # 测试集比例
    RANDOM_STATE = 42            # 随机种子
    TFIDF_PARAMS = {
        'max_df': 0.8,           # 最多出现在80%文档中
        'min_df': 3              # 最少在3个文档中出现
    }
    LOGISTIC_REGRESSION_PARAMS = {
        'max_iter': 1000,        # 最大迭代次数
        'C': 1.0                 # 正则化参数
    }
```

---

## 📝 使用示例

### 示例 1: 完整训练流程

```bash
cd src
python main.py
```

### 示例 2: 多模型对比实验

打开 Jupyter Notebook 进行多模型对比：

```bash
# 启动 Jupyter Notebook
jupyter notebook notebooks/exploration.ipynb
```

notebook 包含以下内容：
- 📊 6 种机器学习模型对比
- 🎯 TF-IDF 和 Count 特征对比
- 📈 准确率、训练时间、过拟合分析
- 🖼️ 性能对比可视化图表
- 📋 详细的混淆矩阵分析
- 💾 模型对比结果 CSV 导出

**对比的模型：**
1. 逻辑回归 (Logistic Regression) - TF-IDF
2. K-最近邻 (KNN) - TF-IDF
3. 随机森林 (Random Forest) - Count
4. 决策树 (Decision Tree) - TF-IDF
5. 朴素贝叶斯 (Naive Bayes) - Count
6. 支持向量机 (SVM) - Count

### 示例 3: 预测单条文本

```python
from src.predict import SentimentPredictor

predictor = SentimentPredictor()

test_texts = [
    "这个电影真是太精彩了，演员演技很棒！",
    "产品质量很差，用了一次就坏了",
    "还行吧，没什么特别的感觉"
]

for text in test_texts:
    result = predictor.predict(text)
    print(f"文本: {result['text']}")
    print(f"情感: {result['sentiment']} (置信度: {result['confidence']}%)")
    print()
```

**输出示例：**
```
文本: 这个电影真是太精彩了，演员演技很棒！
情感: Positive (置信度: 92.45%)

文本: 产品质量很差，用了一次就坏了
情感: Negative (置信度: 87.63%)

文本: 还行吧，没什么特别的感觉
情感: Negative (置信度: 65.22%)
```

---

## 🎯 项目目标

- [x] 构建完整的情感分析流程
- [x] 实现数据清洗和预处理
- [x] 训练高准确率的分类模型
- [x] 实现多模型对比实验
- [x] 提供易用的预测接口
- [x] 编写单元测试
- [x] 完善项目文档

---

## 🔧 已知问题与解决方案

### 问题 1: KNN 和 SVM 模型性能较差

**现象：**
- KNN 准确率约 43%（远低于其他模型）
- SVM 准确率约 65%（显著低于逻辑回归的 96%）

**根本原因：**
- 特征提取产生的是**稀疏矩阵**（sparse matrix）
- KNN 和 SVM 等基于距离的算法在高维稀疏数据上表现不佳
- 稀疏矩阵在 scikit-learn 中的距离计算效率低，精度有限

**解决方案：**
在 `exploration.ipynb` 中，我们对这两个模型进行了优化：

```python
# KNN 模型优化
X_train_tfidf_dense = X_train_tfidf.toarray()  # 转换为稠密矩阵
X_test_tfidf_dense = X_test_tfidf.toarray()

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_tfidf_dense, y_train)

# SVM 模型优化
X_train_count_dense = X_train_count.toarray()  # 转换为稠密矩阵
X_test_count_dense = X_test_count.toarray()

svm_model = SVC(kernel='rbf', probability=True)
svm_model.fit(X_train_count_dense, y_train)
```

**优化效果：**
- ✅ 将稀疏矩阵转换为稠密矩阵后，距离计算更准确
- ✅ 模型计算精度显著提升
- ⚠️ 注意：稠密矩阵会占用更多内存，大规模数据集需要谨慎使用

**建议：**
1. **对于小到中等规模数据集**：建议使用优化后的 KNN 和 SVM
2. **对于大规模数据集**：推荐使用 Logistic Regression 或 Random Forest
3. **生产环境**：优先选择 Logistic Regression（准确率 96.30% + 低延迟 0.046s）

### 问题 2: 文本清理的一致性

**解决方案：**
所有涉及文本预测的模块都进行了统一的文本清理：
- 使用 `src/data/cleaner.py` 中的 `clean_text()` 函数
- 确保 `analysis.py` 和 `predict.py` 的预测结果一致
- 详见 `tests/test_consistency.py` 中的验证测试

---

##  未来改进方向

- [ ] 集成更多机器学习模型 (XGBoost, LightGBM, etc.)
- [ ] 集成深度学习模型 (CNN, RNN, BERT, ERNIE)
- [ ] 增加情感强度评分（不仅是正/负）
- [ ] 支持多标签分类
- [ ] 构建 Web API 服务 (Flask/FastAPI)
- [ ] 模型蒸馏和压缩
- [ ] 增量学习能力
- [ ] 自动化超参数调优 (Optuna, Ray Tune)
- [ ] 性能基准测试和优化
- [ ] 支持多语言情感分析
