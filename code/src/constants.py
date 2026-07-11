import os

# Constants
NUM_CLASSES_MNIST = 10
NUM_CLASSES_CIFAR10 = 10

# Keys
ACCURACY_KEY = "accuracy"

# Paths
# Anchor to this file's location (src/) so the dataset download dir always
# resolves to code/data/, regardless of the cwd a script is invoked from.
DATA_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))