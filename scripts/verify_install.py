import torch
import torchvision
import numpy
import pandas
import sklearn
import PIL
import yaml
import matplotlib
import seaborn

print(f"PyTorch:      {torch.__version__}")
print(f"Torchvision:  {torchvision.__version__}")
print(f"CUDA:         {torch.cuda.is_available()}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device:       {device}")
print(f"NumPy:        {numpy.__version__}")
print(f"Pandas:       {pandas.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print(f"Pillow:       {PIL.__version__}")
print("---")
print("All imports OK")
