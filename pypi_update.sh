# make changes to setup.py and others first.
# update setuptools, wheel
python3 -m pip install --user --upgrade setuptools wheel
# remove previous set up
rm dist/*
# set up
python3 setup.py sdist bdist_wheel
# bunch of stuff comes out
# update twine
python3 -m pip install --user --upgrade twine
# twine upload
python3 -m twine upload dist/*
