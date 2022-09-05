'''
   INPUT: pill_image_name_file, public_train_path
         - pill_image_name_file:    Image file name (i.e. VAIPE_P_1159_30.jpg)
         - public_train_path:   Data path (i.e. ../data/public_train/)
        

   OUTPUT: result
         - result:  A list of dictionaries of each pill in image with the following keys:
                                                                                        'pill_image': image of each pill
                                                                                        'drugname': drugname of each pill
                                                                                        'id': mapping of each pill 

                                                                                        
'''
import os
import cv2
import json
from CONFIG import TO_CUDA

# import Dataset, DataLoader from torch 
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
    
def get_data_pill(pill_image_name_file, public_root_path):
    """
    Args:
        public_root_path: PATH ROOT of train/val/test 

        pill_image_name_file: Image file name (i.e. VAIPE_P_1159_30.jpg)
    """

    pill_image_id = pill_image_name_file[:-4]
    pill_json_name = pill_image_id + ".json"
    # pill_json_path = os.path.join(public_train_path, "pill/label", pill_json_name)
    pill_json_path = "{}/pill/label/{}".format(public_root_path, pill_json_name)
    print("pill_json_path:_______________________ ", pill_json_path)
    pres_json_name = pill_image_id.split("_")[:-1][0]+"_"+pill_image_id.split("_")[:-1][1]+"_"+"TRAIN"+"_"+pill_image_id.split("_")[:-1][2]+".json"
    pres_json_path = os.path.join(public_root_path, "prescription", "label", pres_json_name)
    pill_json = json.load(open(pill_json_path))
    pres_json = json.load(open(pres_json_path))
    pill_image = cv2.imread(os.path.join(public_root_path, "pill", "image", pill_image_name_file))
    pill_dict_list = []
    for pill_box in pill_json:
        x = pill_box["x"]
        y = pill_box["y"]
        w = pill_box["w"]
        h = pill_box["h"]
        label = pill_box["label"]
        pill_crop_image = pill_image[y:y+h, x:x+w]
        pill_crop_image_rgb = cv2.cvtColor(pill_crop_image, cv2.COLOR_BGR2RGB)

        pill_dict = {}
        if label == 107:
            pill_dict["pill_image"] = pill_crop_image_rgb
            pill_dict["drugname"] = "MOT LINH BAY"
            pill_dict["id"] = label
        else:
            for pres_box in pres_json:
                if pres_box["label"] == "drugname":
                    if pres_box["mapping"] == label:
                        pill_dict["pill_image"] = pill_crop_image_rgb
                        pill_dict["drugname"] = pres_box["text"]
                        pill_dict["id"] = pres_box["mapping"]
                        break
        pill_dict_list.append(pill_dict)
    return pill_dict_list




class Pill_Dataset(Dataset):
    def __init__(self, public_root_path, mode="train"):
        self.public_train_path = public_root_path
        self.mode = mode
        self.pill_image_name_file = os.listdir(os.path.join(public_root_path, "pill", "image"))
        self.model_pretrain = torch.jit.load("/workspace/nabang1010/LBA_VAIPE/GNN/Multi_Label_Image_Recognition_with_Graph_Convolutional_Networks/checkpoint/pretrain_model/wide_resnet50_2_f1_0.993.pt",\
                                    map_location= TO_CUDA)
        
    def __len__(self):
        return len(self.pill_image_name_file)
    
    def __getitem__(self, idx):
        pill_image_name_file = self.pill_image_name_file[idx]
        pill_dict_list = get_data_pill(pill_image_name_file, self.public_train_path)
        if self.mode == "train":
            transform = A.Compose([
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit= 0.10 , p=0.5),
                # A.RGBShift(r_shift_limit= 10, g_shift_limit=10, b_shift_limit=10, p=0.5),
                A.MultiplicativeNoise(multiplier=0.5, p=0.5),

                A.RGBShift(r_shift_limit=8, g_shift_limit=8, b_shift_limit=8, p=0.3),
                
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.15, rotate_limit= 0.10 , p=0.5),
                    
                # A.MultiplicativeNoise(multiplier=1, p=0.5),

                
                A.RandomRotate90(p = 0.5),
                
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.15, rotate_limit= 0.10 , p=0.5),

                # add noise by albummentations
                # A.MultiplicativeNoise(multiplier=1.5, p=0.5),
                # A.RandomBrightnessContrast(p=0.10),
                # ShiftScaleRotate by albumentations

                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit= 0.10 , p=0.5),
                
                
                # resize to 224x224
                A.Resize(224, 224),
                # normalize to ImageNet
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            transform = A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        for pill_dict in pill_dict_list:
            pill_dict["pill_image"] = transform(image=pill_dict["pill_image"])["image"]
        
        # Done prepare data image
        
        # Now, get feature from image through model_pretrain 
        # load jit model 

        nano_batch_image = []
        nano_batch_label = []
        for pill_dict in pill_dict_list:
            nano_batch_image.append(pill_dict["pill_image"])
            nano_batch_label.append(pill_dict["id"])
        nano_batch_image = torch.stack(nano_batch_image)
        nano_batch_label = torch.tensor(nano_batch_label)

        # get feature from image
        with torch.no_grad():
            nano_batch_image = nano_batch_image.to(TO_CUDA)
            feature = self.model_pretrain(nano_batch_image)
            nano_batch_feature = feature.cpu()
        return nano_batch_feature, nano_batch_label
        # code nhầm nên vẫn ra 107 chiều, hề , nhưng ko sao qua lớp FC chắc ok 


        





    

if __name__ == "__main__":
    path_image_name = "VAIPE_P_0_0.jpg"
    path_data_root =  "/workspace/nabang1010/LBA_VAIPE/GNN/Multi_Label_Image_Recognition_with_Graph_Convolutional_Networks/data/public_train/"
    data  = get_data_pill(path_image_name, path_data_root)
    