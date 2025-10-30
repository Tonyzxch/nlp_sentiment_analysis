# 安装脚本
from setuptools import setup, find_packages

setup(
    name="nlp-sentiment-analysis",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "jieba>=0.42.1",
        "pandas>=1.5.0",
        "scikit-learn>=1.2.0",
        "numpy>=1.24.0"
    ],
    python_requires=">=3.8",
)