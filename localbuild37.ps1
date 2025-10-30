# 激活 python37
$baseDir = 'D:\GreenSoft\python-3.7.0-embed-win32' # amd64-> win32 少10M
$env:PATH = "$baseDir;$baseDir\Scripts;" + $env:PATH

# 安装

#python -m pip install . 
python -m pip install . --no-deps



# 保持窗口一段时间，或者等待用户按键


$timer=Start-Job {Start-Sleep 20}; Write-Host "Press any key to continue..."; while(-not [console]::KeyAvailable -and (Get-Job -Id $timer.Id).State -eq 'Running'){Start-Sleep 0.1}; Stop-Job $timer