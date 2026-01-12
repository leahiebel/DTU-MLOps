from cookiecutter_project.model import MyAwesomeModel
import torch

# tests/test_model.py
import pytest
from cookiecutter_project.model import MyAwesomeModel

def test_error_on_wrong_shape():
    model = MyAwesomeModel()
    with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
        model(torch.randn(1,2,3))
    with pytest.raises(ValueError, match=r"Expected each sample to have shape \[1, 28, 28\]"):
        model(torch.randn(1,1,28,29))