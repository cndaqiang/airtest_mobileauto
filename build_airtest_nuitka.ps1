# ============================================================
# build_airtest_nuitka.ps1 - 编译 airtest_mobileauto 为 .pyd
# ============================================================
#
# 简单方案：
#   1. 正常安装 .py 版本
#   2. 编译 .pyd
#   3. 删除 site-packages 中的 .py 和 .pyc
#   4. 复制 .pyd 到 site-packages
#
# ============================================================

# 激活 Python 3.7 32位环境
$baseDir = 'D:\GreenSoft\WPy-3702\python-3.7.0'
$sitePackages = "$baseDir\Lib\site-packages\airtest_mobileauto"
$env:PATH = "$baseDir;$baseDir\Scripts;" + $env:PATH
$env:PYTHONIOENCODING = "utf-8"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  编译 airtest_mobileauto 为 .pyd" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# ============================================================
# 步骤 1: 正常安装
# ============================================================
Write-Host "=== 步骤 1: 正常安装 ===" -ForegroundColor Yellow
python -m pip install . --no-deps 2>&1 | Out-Null
Write-Host "✓ 安装完成" -ForegroundColor Green
Write-Host ""

# ============================================================
# 步骤 2: 编译 .pyd
# ============================================================
Write-Host "=== 步骤 2: 编译 .pyd ===" -ForegroundColor Yellow
cd airtest_mobileauto

Write-Host "  → 编译 control.py ..." -ForegroundColor Gray
python -m nuitka --module control.py --mingw64 --assume-yes-for-downloads --quiet 2>&1 | Out-Null

Write-Host "  → 编译 pick2yaml.py ..." -ForegroundColor Gray
python -m nuitka --module pick2yaml.py --mingw64 --assume-yes-for-downloads --quiet 2>&1 | Out-Null

if ((Test-Path "control.cp37-win32.pyd") -and (Test-Path "pick2yaml.cp37-win32.pyd")) {
    Write-Host "✓ 编译完成" -ForegroundColor Green
} else {
    Write-Host "✗ 编译失败" -ForegroundColor Red
    cd ..
    pause
    exit 1
}

cd ..
Write-Host ""

# ============================================================
# 步骤 3: 删除 site-packages 中的 .py 和 .pyc
# ============================================================
Write-Host "=== 步骤 3: 删除 .py 和 .pyc 文件 ===" -ForegroundColor Yellow
Remove-Item "$sitePackages\control.py" -Force
Remove-Item "$sitePackages\pick2yaml.py" -Force
Remove-Item "$sitePackages\__pycache__\control*.pyc" -Force -ErrorAction SilentlyContinue
Remove-Item "$sitePackages\__pycache__\pick2yaml*.pyc" -Force -ErrorAction SilentlyContinue
Write-Host "✓ 清理完成" -ForegroundColor Green
Write-Host ""

# ============================================================
# 步骤 4: 复制 .pyd 到 site-packages
# ============================================================
Write-Host "=== 步骤 4: 复制 .pyd 到 site-packages ===" -ForegroundColor Yellow
Copy-Item "airtest_mobileauto\control.cp37-win32.pyd" "$sitePackages\control.pyd" -Force
Copy-Item "airtest_mobileauto\pick2yaml.cp37-win32.pyd" "$sitePackages\pick2yaml.pyd" -Force
Write-Host "✓ 复制完成" -ForegroundColor Green
Write-Host ""

# ============================================================
# 验证
# ============================================================
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  完成！" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "安装位置: $sitePackages" -ForegroundColor Cyan
Write-Host ""
Write-Host "代码保护状态：" -ForegroundColor Yellow
Write-Host "  [⭐⭐⭐⭐⭐] control.pyd" -ForegroundColor Green
Write-Host "  [⭐⭐⭐⭐⭐] pick2yaml.pyd" -ForegroundColor Green
Write-Host ""
Write-Host "验证导入：" -ForegroundColor Yellow
python -c "import airtest_mobileauto; print('✓ 导入成功'); print('位置:', airtest_mobileauto.__file__)"
python -c "from airtest_mobileauto import control; print('✓ control 导入成功')"
Write-Host ""
Write-Host "现在可以运行 autowzry 的 buildexe37.ps1" -ForegroundColor Cyan
Write-Host ""

# ============================================================
# 清理临时文件（可选）
# ============================================================
# 取消注释以下代码可自动清理编译产物（保持项目目录整洁）

Write-Host "提示：如需清理编译产物，可取消注释脚本末尾的清理代码" -ForegroundColor DarkGray
Write-Host ""

# # 进入源码目录清理
# cd airtest_mobileauto
#
# Write-Host "清理编译产物..." -ForegroundColor Yellow
#
# # Nuitka 编译产物
# Remove-Item "control.build" -Recurse -Force -ErrorAction SilentlyContinue
# Remove-Item "pick2yaml.build" -Recurse -Force -ErrorAction SilentlyContinue
#
# # .pyd 文件（原始版本）
# Remove-Item "control.cp37-win32.pyd" -Force -ErrorAction SilentlyContinue
# Remove-Item "pick2yaml.cp37-win32.pyd" -Force -ErrorAction SilentlyContinue
#
# # .pyi 类型存根文件（如果生成）
# Remove-Item "*.pyi" -Force -ErrorAction SilentlyContinue
#
# cd ..
#
# Write-Host "✓ 清理完成" -ForegroundColor Green
# Write-Host ""

Write-Host "按任意键退出..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
