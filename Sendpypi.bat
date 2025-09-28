@echo off
REM 删除 dist 目录
rmdir /s /q dist

REM 升级构建和上传工具
%USERPROFILE%\AppData\Local\anaconda3\python.exe -m pip install --upgrade build twine

REM 使用 PEP 517 构建 (会生成 sdist 和 wheel)
%USERPROFILE%\AppData\Local\anaconda3\python.exe -m build

REM 上传到 PyPI
%USERPROFILE%\AppData\Local\anaconda3\python.exe -m twine upload dist/*
