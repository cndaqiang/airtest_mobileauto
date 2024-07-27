python setup.py sdist
python -m pip install .\dist\airtest_mobileauto-1.3.4.tar.gz
twine upload dist/*
