rm .\dist\*
python setup.py sdist
python -m pip install .\dist\airtest_mobileauto-1.3.5.1.tar.gz
#twine upload dist/*
