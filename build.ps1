& 'C:\Users\cnche\AppData\Local\anaconda3\shell\condabin\conda-hook.ps1'
conda activate 'C:\Users\cnche\AppData\Local\anaconda3'
Get-ChildItem -Path .\dist\* | Remove-Item
python setup.py sdist
Get-ChildItem -Path .\dist\* | ForEach-Object { python -m pip install $_.FullName }
#twine upload dist/*
