# Setup virtual environment

## create virtual environment
python -m venv pydata

## activate it
source ./pydata/bin/activate

## install requirements
pip install -r requirements.txt 

## start jupyter
jupyter notebook

## Fasttext Autotune

If you want to use the fasttext built-in hyperparameter tuning install fasttext directly from git repository.

```bash
$ git clone https://github.com/facebookresearch/fastText.git
$ cd fastText
$ sudo pip install .
$ # or :
$ sudo python setup.py install
```
