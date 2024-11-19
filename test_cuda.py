import torch

# Vérifie si CUDA est disponible
if torch.cuda.is_available():
    print(f"CUDA est disponible. Version CUDA : {torch.version.cuda}")
    print(f"cuDNN est disponible. Version cuDNN : {torch.backends.cudnn.version()}")
else:
    print("CUDA n'est pas disponible.")
