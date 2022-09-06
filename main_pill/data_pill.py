import torch 
# import Dataset, DataLoader, and other utilities
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class Data_Pill(Dataset):
    def __init__(self, data_csv_full, data_csv_just_concat):
        self.data_full = pd.read_csv(data_csv_full)
        self.data_emd = pd.read_csv(data_csv_just_concat)
        # load model by jit or torchscript
        self.model_base = torch.jit.load('/workspace/nabang1010/LBA_VAIPE/GNN/Multi_Label_Image_Recognition_with_Graph_Convolutional_Networks/checkpoint/pretrain_model/wide_resnet50_2_f1_0.993.pt')
        self.model_base = self.model_base.to('cuda')
    def __len__(self):
        return len(self.data_csv)

    def __getitem__(self, idx):
        data_full_line = self.data_full.iloc[idx]
        data_emd_line  = self.data_emb[self.data_emb["pres_file_name"] == data_full_line["pres_file_name"]]
        drugname_embedding =  eval(data_emd_line["drugname_embedding"].values[0])
        diagnose_embedding =  eval(data_emd_line["diagnose_embedding"].values[0])
        quantity_embedding =  eval(data_emd_line["quantity_embedding"].values[0])

        # transform to tensor
        drugname_embedding = torch.tensor(drugname_embedding)
        diagnose_embedding = torch.tensor(diagnose_embedding)
        quantity_embedding = torch.tensor(quantity_embedding)

        # get image 
        image_folder = str(data_full_line["image"])
        if image_folder != "107"
            path_all_folder = "/workspace/nabang1010/LBA_VAIPE/GNN/Multi_Label_Image_Recognition_with_Graph_Convolutional_Networks/data/pill_classify/data_classification_aug"
            path_image = os.path.join(path_all_folder, image_folder)
            




        return drugname_embedding, diagnose_embedding, quantity_embedding


