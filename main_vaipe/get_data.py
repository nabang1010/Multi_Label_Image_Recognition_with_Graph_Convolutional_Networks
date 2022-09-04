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


def get_data_pill(pill_image_name_file, public_train_path):
    pill_image_id = pill_image_name_file[:-4]
    pill_json_name = pill_image_id + ".json"
    pill_json_path = os.path.join(public_train_path, "pill", "label", pill_json_name)
    pres_json_name = pill_image_id.split("_")[:-1][0]+"_"+pill_image_id.split("_")[:-1][1]+"_"+"TRAIN"+"_"+pill_image_id.split("_")[:-1][2]+".json"
    pres_json_path = os.path.join(public_train_path, "prescription", "label", pres_json_name)
    pill_json = json.load(open(pill_json_path))
    pres_json = json.load(open(pres_json_path))
    pill_image = cv2.imread(os.path.join(public_train_path, "pill", "image", pill_image_name_file))
    result = []
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
        result.append(pill_dict)
    return result


if __name__ == "__main__":
    get_data_pill()