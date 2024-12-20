rmdir /s /q dist
%USERPROFILE%\AppData\Local\anaconda3\python.exe setup.py sdist
%USERPROFILE%\AppData\Local\anaconda3\python.exe -m twine upload dist/*
