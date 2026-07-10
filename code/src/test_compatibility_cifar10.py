import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from models.cifar10_cnn import Cifar10ResNet20

model = Cifar10ResNet20()
errors = ModuleValidator.validate(model, strict=False)
if errors:
    print('Incompatible layers:', errors)
else:
    print('Model is fully compatible with Opacus!')
