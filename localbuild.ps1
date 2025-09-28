# 初始化 conda 环境
& "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"
conda activate "$env:USERPROFILE\miniconda3"

# 清理旧构建文件
Remove-Item -Path .\dist\* -Force -ErrorAction SilentlyContinue

# 构建 sdist 和 wheel (推荐同时生成)
python -m pip install --upgrade build twine
python -m build

# 安装生成的包（wheel 优先）
Get-ChildItem -Path .\dist\* | ForEach-Object { python -m pip install --force-reinstall $_.FullName }

# 上传到 PyPI (如果需要)
# python -m twine upload dist/*
