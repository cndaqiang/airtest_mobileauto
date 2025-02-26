from setuptools import setup, find_packages

setup(
    name='airtest_mobileauto',
    version='2.1.7',
    author='cndaqiang',
    author_email='who@cndaqiang.ac.cn',
    description='A robust, object-oriented, multi-process mobile app control framework based on AirTest, designed for stable and compatible debugging and automation of devices and apps. Ideal for tasks such as game automation in titles like Honor of Kings, with enhanced stability features including connection checks, automatic retries on failure, and automatic restarts for continuous operation.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    package_data={
        # 假设 tpl_target_pos.png 位于 'airtest_mobileauto' 包的根目录下
        'airtest_mobileauto': ['tpl_target_pos.png'],
    },
    include_package_data=True,  # 确保 package_data 里的文件被包含
    url='https://github.com/cndaqiang/airtest_mobileauto', 
    install_requires=[
        'airtest',
        'numpy',
        'pyyaml',
    ],
    extras_require={
        'windows': ['pypiwin32'],  # Windows 系统需要的额外依赖
        # macOS 和 Linux 系统没有额外依赖，所以这里可以不写
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Utilities',
    ],
    python_requires='>=3.6',
)