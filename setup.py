from setuptools import setup, find_packages

setup(
    name='airtest_mobileauto',
    version='1.3.5',
    author='cndaqiang',
    author_email='who@cndaqiang.ac.cn',
    description='A robust, object-oriented, multi-process mobile app control framework based on AirTest, designed for stable and compatible debugging and automation of devices and apps. Ideal for tasks such as game automation in titles like Honor of Kings, with enhanced stability features including connection checks, automatic retries on failure, and automatic restarts for continuous operation.', 
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    url='https://github.com/cndaqiang/airtest_mobileauto',
    install_requires=[
        'airtest',
        'numpy',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Utilities',
    ],
    python_requires='>=3.6',
)