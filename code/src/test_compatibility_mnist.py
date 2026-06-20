import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from models.mnist_cnn import MnistCNN

model = MnistCNN()
errors = ModuleValidator.validate(model, strict=False)
if errors:
    print('Incompatible layers:', errors)
else:
    print('Model is fully compatible with Opacus!')