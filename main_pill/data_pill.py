import torch 
# import Dataset, DataLoader, and other utilities
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os 
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


class Data_Pill_random_all(Dataset):
    def __init__(self, data_full, data_emb, status):
        self.data_full = data_full
        self.data_emb = data_emb
        self.model_base = torch.jit.load('/workspace/nabang1010/LBA_VAIPE/GNN/Multi_Label_Image_Recognition_with_Graph_Convolutional_Networks/checkpoint/pretrain_model/wide_resnet50_2_f1_0.993.pt',\
             map_location='cuda')
        self.status = status
    def __len__(self):
        return len(self.data_full)

    def __getitem__(self, idx):
        data_full_line = self.data_full.iloc[idx]
        data_emd_line  = self.data_emb[self.data_emb["pres_file_name"] == data_full_line["pres_file_name"]]
        drugname_embedding =  eval(data_emd_line["drugname_embedding"].values[0])
        diagnose_embedding =  eval(data_emd_line["diagnose_embedding"].values[0])
        quantity_embedding =  eval(data_emd_line["quantity_embedding"].values[0])

        # transform to tensor
        drugname_embedding = torch.tensor(drugname_embedding, dtype=torch.float32)
        diagnose_embedding = torch.tensor(diagnose_embedding, dtype=torch.float32)
        quantity_embedding = torch.tensor(quantity_embedding, dtype=torch.float32)

        # transform image to ImageNet type

        transform_albumentations = A.Compose([
            # resize 224x224
            A.Resize(224, 224),
            # normalize
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # to tensor
            ToTensorV2()
        ])



        # get image 
        image_folder = str(data_full_line["mapping"])
        if image_folder != "107":
            path_all_folder = "/workspace/nabang1010/LBA_VAIPE/GNN/Multi_Label_Image_Recognition_with_Graph_Convolutional_Networks/data/pill_classify/data_classification_aug"
            path_image_folder = os.path.join(path_all_folder, image_folder)
            image_list = os.listdir(path_image_folder)
            #random select image
            image_name = random.choice(image_list)
            path_image = os.path.join(path_image_folder, image_name)
            image_cv2 = cv2.imread(path_image)
            # transform image by albumentations
            image_albumentations = transform_albumentations(image=image_cv2)["image"]
            # get image embedding
            with torch.no_grad():
                image_embedding = self.model_base(image_albumentations.unsqueeze(0).to('cuda'))
        else:
                #################
                # fix code here #
                #################


            path_all_folder = "/workspace/nabang1010/LBA_VAIPE/GNN/Multi_Label_Image_Recognition_with_Graph_Convolutional_Networks/data/pill_classify/data_classification_aug"
            # random a folder
            folder_list = os.listdir(path_all_folder)
            if "107" in folder_list:
                folder_list.remove("107")
            folder_name = random.choice(folder_list)
            path_image_folder = os.path.join(path_all_folder, folder_name)
            image_list = os.listdir(path_image_folder)
            #random select image
            image_name = random.choice(image_list)
            path_image = os.path.join(path_image_folder, image_name)
            image_cv2 = cv2.imread(path_image)
            # transform image by albumentations
            image_albumentations = transform_albumentations(image=image_cv2)["image"]
            # get image embedding
            with torch.no_grad():
                image_embedding = self.model_base(image_albumentations.unsqueeze(0).to('cuda'))


        label = torch.tensor(data_full_line["mapping"])
        return image_embedding, drugname_embedding, diagnose_embedding, quantity_embedding, label






class Data_Pill_107_raw(Dataset):
    '''
    Class này lựa chọn 107 ngẫu nhiên và chưa áp dụng luật để bỏ những class tương tự class đang xét 
    '''
    def __init__(self, data_full, data_emb, status = ""):
        self.data_full = data_full
        self.data_emb = data_emb
        self.model_base = torch.jit.load('/workspace/nabang1010/LBA_VAIPE/GNN/Multi_Label_Image_Recognition_with_Graph_Convolutional_Networks/checkpoint/pretrain_model/wide_resnet50_2_f1_0.993.pt',\
             map_location='cuda')
        self.status = status
        
    def __len__(self):
        return len(self.data_full)

    def __getitem__(self, idx):

        data_full_line = self.data_full.iloc[idx]
        data_emd_line  = self.data_emb[self.data_emb["pres_file_name"] == data_full_line["pres_file_name"]]
        drugname_embedding =  eval(data_emd_line["drugname_embedding"].values[0])
        diagnose_embedding =  eval(data_emd_line["diagnose_embedding"].values[0])
        quantity_embedding =  eval(data_emd_line["quantity_embedding"].values[0])

        # transform to tensor
        drugname_embedding = torch.tensor(drugname_embedding, dtype=torch.float32)
        diagnose_embedding = torch.tensor(diagnose_embedding, dtype=torch.float32)
        quantity_embedding = torch.tensor(quantity_embedding, dtype=torch.float32)

        
        
        # transform image to ImageNet type
        transform_albumentations = A.Compose([
            # resize 224x224
            A.Resize(224, 224),
            # normalize
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # to tensor
            ToTensorV2()
        ])

        image_folder = str(data_full_line["mapping"])
        
        if self.status == "train":
            '''
            Trạng thái train và valid sẽ khác nhau ở  path_all_folder: đường dẫn tới tập dữ liệu 
            '''
            if image_folder != "107":
                # get image 
                
                ############################################
                #Thay thế đường dẫn từ tập chưa fix thành tập fixed (tập 1000) ở dưới này 
                ###############################################

                path_all_folder = "/workspace/nabang1010/LBA_VAIPE/GNN/Multi_Label_Image_Recognition_with_Graph_Convolutional_Networks/data/pill_classify/data_classification_fixed_aug"
                path_image_folder = os.path.join(path_all_folder, image_folder)
                image_list = os.listdir(path_image_folder)
                #random select image
                image_name = random.choice(image_list)
                path_image = os.path.join(path_image_folder, image_name)
                image_cv2 = cv2.imread(path_image)
                # transform image by albumentations
                image_albumentations = transform_albumentations(image=image_cv2)["image"]
                # get image embedding
                with torch.no_grad():
                    image_embedding = self.model_base(image_albumentations.unsqueeze(0).to('cuda'))
                

            if image_folder == "107":
                #################
                # fix code here #
                #################
                train_pres_path = "/workspace/nabang1010/LBA_VAIPE/GNN/Multi_Label_Image_Recognition_with_Graph_Convolutional_Networks/data/pill_classify/pill_vs_pres/train"
                pres_name = data_full_line["pres_file_name"]
                pres_path = os.path.join(train_pres_path, pres_name)
                pres_pill_list = os.listdir(pres_path)

                if "107" in pres_pill_list:
                    image_107_folder = os.path.join(pres_path, "107")
                    image_107_list = os.listdir(image_107_folder)
                    image_107_name = random.choice(image_107_list)
                    image_107_path = os.path.join(image_107_folder, image_107_name)
                    image_107_cv2 = cv2.imread(image_107_path)
                    image_107_albumentations = transform_albumentations(image=image_107_cv2)["image"]
                    ##############################################################
                    ### Phải augment thằng 107 ở đây, để tạo đa dạng qua mỗi epoch
                    ##############################################################
                    with torch.no_grad():
                        image_embedding = self.model_base(image_107_albumentations.unsqueeze(0).to('cuda'))
                else:
                    ############################################################
                    #### CODE PHẦN NÀY SAU KHI XỬ LÝ ĐƯỢC FILE CSV RÀNG BUỘC
                    #### Phần này sẽ lựa chọn ngẫu nhiên 1 trong 106 class thoả mãn ràng buộc 
                    ############################################################

                    pass
                    

        if self.status == "val":
            '''
            Trạng thái train và valid sẽ khác nhau ở  path_all_folder: đường dẫn tới tập dữ liệu 
            
            File csv valid sẽ khác với file csv train ở chỗ:
                Nhãn 107 sẽ có cụ thể cho hoá đơn chứ không phải tất cả
            '''
            
            val_pres_path = "/workspace/nabang1010/LBA_VAIPE/GNN/Multi_Label_Image_Recognition_with_Graph_Convolutional_Networks/data/pill_classify/pill_vs_pres/val"
            pres_name = data_full_line["pres_file_name"]
            pres_path = os.path.join(val_pres_path, pres_name)
            pres_pill_list = os.listdir(pres_path)

            if image_folder in pres_pill_list:
                image_folder_path = os.path.join(pres_path, image_folder)
                image_list = os.listdir(image_folder_path)
                image_name = random.choice(image_list)
                image_path = os.path.join(image_folder_path, image_name)
                image_cv2 = cv2.imread(image_path)
                image_albumentations = transform_albumentations(image=image_cv2)["image"]
                with torch.no_grad():
                    image_embedding = self.model_base(image_albumentations.unsqueeze(0).to('cuda'))
            
        label = torch.tensor(data_full_line["mapping"])
        return image_embedding, drugname_embedding, diagnose_embedding, quantity_embedding, label 


                

 


class Data_Pill_107_fix(Dataset):
    '''
    Class này lựa chọn 107 ngẫu nhiên và chưa áp dụng luật để bỏ những class tương tự class đang xét 
    '''
    def __init__(self, data_full, data_emb, status = ""):
        self.data_full = data_full
        self.data_emb = data_emb
        self.model_base = torch.jit.load('/workspace/nabang1010/LBA_VAIPE/GNN/Multi_Label_Image_Recognition_with_Graph_Convolutional_Networks/checkpoint/pretrain_model/wide_resnet50_2_f1_0.993.pt',\
             map_location='cuda')
        self.status = status
        
    def __len__(self):
        return len(self.data_full)

    def __getitem__(self, idx):

        data_full_line = self.data_full.iloc[idx]
        data_emd_line  = self.data_emb[self.data_emb["pres_file_name"] == data_full_line["pres_file_name"]]
        drugname_embedding =  eval(data_emd_line["drugname_embedding"].values[0])
        diagnose_embedding =  eval(data_emd_line["diagnose_embedding"].values[0])
        quantity_embedding =  eval(data_emd_line["quantity_embedding"].values[0])

        # transform to tensor
        drugname_embedding = torch.tensor(drugname_embedding, dtype=torch.float32)
        diagnose_embedding = torch.tensor(diagnose_embedding, dtype=torch.float32)
        quantity_embedding = torch.tensor(quantity_embedding, dtype=torch.float32)

        
        
        # transform image to ImageNet type
        transform_albumentations = A.Compose([
            # resize 224x224
            A.Resize(224, 224),
            # normalize
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # to tensor
            ToTensorV2()
        ])

        image_folder = str(data_full_line["mapping"]) # tên của lọai thuốc
        
        if self.status == "train":
            df_thay_the = pd.read_csv("/workspace/nabang1010/LBA_VAIPE/GNN/Multi_Label_Image_Recognition_with_Graph_Convolutional_Networks/data/pill_classify/thuoc_thay_the.csv") 
            '''
            Trạng thái train và valid sẽ khác nhau ở  path_all_folder: đường dẫn tới tập dữ liệu 
            '''
            if image_folder != "107":
                # get image 
                
                ############################################
                #Thay thế đường dẫn từ tập chưa fix thành tập fixed (tập 1000) ở dưới này  ==> DONE
                ###############################################

                path_all_folder = "/workspace/nabang1010/LBA_VAIPE/GNN/Multi_Label_Image_Recognition_with_Graph_Convolutional_Networks/data/pill_classify/data_classification_fixed_aug"
                path_image_folder = os.path.join(path_all_folder, image_folder)
                image_list = os.listdir(path_image_folder)
                #random select image
                image_name = random.choice(image_list)
                path_image = os.path.join(path_image_folder, image_name)
                image_cv2 = cv2.imread(path_image)
                # transform image by albumentations
                image_albumentations = transform_albumentations(image=image_cv2)["image"]
                # get image embedding
                with torch.no_grad():
                    image_embedding = self.model_base(image_albumentations.unsqueeze(0).to('cuda'))
                

            if image_folder == "107":
                #################
                # fix code here # ==> Done
                #################
                train_pres_path = "/workspace/nabang1010/LBA_VAIPE/GNN/Multi_Label_Image_Recognition_with_Graph_Convolutional_Networks/data/pill_classify/pill_vs_pres/train"
                pres_name = data_full_line["pres_file_name"]
                pres_path = os.path.join(train_pres_path, pres_name)
                pres_pill_list = os.listdir(pres_path)

                transform_train_107 = A.Compose([
                        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit= 0.10 , p=0.5),
                        A.MultiplicativeNoise(multiplier=0.5, p=0.5),

                        A.RGBShift(r_shift_limit=8, g_shift_limit=8, b_shift_limit=8, p=0.3),
                        
                        A.ShiftScaleRotate(
                            shift_limit=0.1, scale_limit=0.15, rotate_limit= 0.10 , p=0.5),
                            
                        # A.MultiplicativeNoise(multiplier=1, p=0.5),

                        
                        A.RandomRotate90(p = 0.5),
                        
                        A.ShiftScaleRotate(
                            shift_limit=0.1, scale_limit=0.1, rotate_limit= 0.10 , p=0.5),

                        A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit= 0.10 , p=0.5),
                    ])

                # random p in [0, 1] 
                p = random.random()
                

                if "107" and p < 0.5 in pres_pill_list: ##### Lát thêm xác suất 50% ở đây ==> DONE
                    image_107_folder = os.path.join(pres_path, "107")
                    image_107_list = os.listdir(image_107_folder)
                    image_107_name = random.choice(image_107_list)
                    image_107_path = os.path.join(image_107_folder, image_107_name)
                    image_107_cv2 = cv2.imread(image_107_path)
                    image_107_albumentations = transform_train_107(image=image_107_cv2)["image"]
                    ##############################################################
                    ### Phải augment thằng 107 ở đây, để tạo đa dạng qua mỗi epoch ==> DONE
                    ##############################################################
                    with torch.no_grad():
                        image_embedding = self.model_base(image_107_albumentations.unsqueeze(0).to('cuda'))
                else:
                    ############################################################
                    #### CODE PHẦN NÀY SAU KHI XỬ LÝ ĐƯỢC FILE CSV RÀNG BUỘC
                    #### Phần này sẽ lựa chọn ngẫu nhiên 1 trong các class thoả mãn ràng buộc ===> DONE
                    ############################################################
                    id_list = [int(i) for i in pres_pill_list]
                    id_list = [i for i in id_list if i != 107]
                    same_all_list = []
                    for id in id_list:
                        same_id_series =  df_thay_the[df_thay_the["drug_id"] == id]["drug_same"]
                        if len(same_id_series) > 0:
                            same_id_list = eval(same_id_series.values[0])
                            same_all_list += same_id_list
                    same_all_list = list(set(same_all_list))
                    same_all_list = [i for i in same_all_list if i != 107]

                    # range list 0 to 106 sub same_all_list
                    range_list = [i for i in range(107)]
                    range_list = [i for i in range_list if i not in same_all_list]

                    # random select id
                    id_random = random.choice(range_list)
                    id_random_folder = str(id_random)
                    # creat path in train folder
                    path_id_random_folder = os.path.join(train_pres_path, id_random_folder)

                    # get image list in id_random_folder
                    image_list = os.listdir(path_id_random_folder)
                    # random select image
                    image_name = random.choice(image_list)
                    path_image_random = os.path.join(path_id_random_folder, image_name)
                    image_cv2_random = cv2.imread(path_image_random)
                    
                    # get image embedding
                    with torch.no_grad():
                        image_embedding = self.model_base(image_cv2_random.unsqueeze(0).to('cuda'))
                    

        if self.status == "val":
            '''
            Trạng thái train và valid sẽ khác nhau ở  path_all_folder: đường dẫn tới tập dữ liệu 
            
            File csv valid sẽ khác với file csv train ở chỗ:
                Nhãn 107 sẽ có cụ thể cho hoá đơn chứ không phải tất cả
            '''
            
            val_pres_path = "/workspace/nabang1010/LBA_VAIPE/GNN/Multi_Label_Image_Recognition_with_Graph_Convolutional_Networks/data/pill_classify/pill_vs_pres/val"
            pres_name = data_full_line["pres_file_name"]
            pres_path = os.path.join(val_pres_path, pres_name)
            pres_pill_list = os.listdir(pres_path)

            if image_folder in pres_pill_list:
                image_folder_path = os.path.join(pres_path, image_folder)
                image_list = os.listdir(image_folder_path)
                image_name = random.choice(image_list)
                image_path = os.path.join(image_folder_path, image_name)
                image_cv2 = cv2.imread(image_path)
                image_albumentations = transform_albumentations(image=image_cv2)["image"]
                with torch.no_grad():
                    image_embedding = self.model_base(image_albumentations.unsqueeze(0).to('cuda'))
            
        label = torch.tensor(data_full_line["mapping"])
        return image_embedding, drugname_embedding, diagnose_embedding, quantity_embedding, label 
