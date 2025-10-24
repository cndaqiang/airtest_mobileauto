# 初始化 conda 环境
& "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"
conda activate "$env:USERPROFILE\miniconda3"

# 清理旧构建文件
Remove-Item -Path .\dist\* -Force -ErrorAction SilentlyContinue

# 构建 sdist 和 wheel
python -m pip install --upgrade build twine
python -m build

# 安装新构建的包
# 切换到脚本所在目录，保证相对路径正确
Set-Location -Path $PSScriptRoot

# 找到 dist 目录下的第一个 .whl
$pkg = Get-ChildItem -Path ".\dist\*.whl" | Select-Object -First 1

if ($null -eq $pkg) {
    Write-Error "No .whl file found in dist directory!"
    exit 1
}

$pkgPath = $pkg.FullName
Write-Host "Installing package: $pkgPath"

python -m pip install --no-deps --no-index --force-reinstall --no-cache-dir "$pkgPath"


# 保持窗口一段时间，或者等待用户按键

Read-Host -Prompt "Press any key to continue . . ."