from setuptools import setup, find_packages

setup(
    name="graduation",  # 库的名称
    version="0.1.0",  # 版本号
    packages=find_packages(),  # 自动发现当前目录下的所有包
    install_requires=[],
    include_package_data=True,  # 包含非代码文件（如数据文件、配置文件等）
)
