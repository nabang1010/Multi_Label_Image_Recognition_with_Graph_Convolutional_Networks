import torch 
TO_CUDA = torch.device("cuda" if torch.cuda.is_available() else "cpu")