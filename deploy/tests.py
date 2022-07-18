from django.test import TestCase

# Create your tests here.

from pathlib import Path

a = Path('./model_data/ml_model.py').resolve()
print(a)


import os
b = os.path.abspath("./model_data/ml_model.sav")
print('B:-', b)

TEST_FILENAME = os.path.join(os.path.dirname(__file__), './model_data/ml_model.sav')

