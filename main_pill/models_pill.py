import torch
import torch.nn as nn


class Model_Concat_Pill(nn.Module):
    def __init__(self, dim_image, dim_diagnose, dim_drugname, dim_quantity):
        super(Model_Concat_Pill, self).__init__()

        self.dim_image = dim_image
        self.dim_diagnose = dim_diagnose
        self.dim_drugname = dim_drugname
        self.dim_quantity = dim_quantity

        # reduce image dim_image
        self.fc_image = nn.Linear(107, dim_image)
        # add norm
        self.norm_image = nn.BatchNorm1d(dim_image)

        # reduce diagnose dim 
        self.fc_diagnose = nn.Linear(768, dim_diagnose*2)
        # add norm
        self.norm_diagnose = nn.BatchNorm1d(dim_diagnose*2)
        self.fc_diagnose_2 = nn.Linear(dim_diagnose*2, dim_diagnose)
        # add norm
        self.norm_diagnose_2 = nn.BatchNorm1d(dim_diagnose)


        # reduce drugname dim
        self.fc_drugname = nn.Linear(768, dim_drugname*2)
        # add norm  
        self.norm_drugname = nn.BatchNorm1d(dim_drugname*2)

        self.fc_drugname_2 = nn.Linear(dim_drugname*2, dim_drugname)
        # add norm
        self.norm_drugname_2 = nn.BatchNorm1d(dim_drugname)


        # add norm quantity
        self.norm_quantity = nn.BatchNorm1d(dim_quantity)
        # dropout 
        self.dropout = nn.Dropout(0.3)
        self.act = nn.Mish()
        

        

        # reduce to 108 class
        # add norm
        self.norm_concat = nn.BatchNorm1d(dim_image + dim_diagnose + dim_drugname + dim_quantity)
        self.fc_concat = nn.Linear(dim_image+dim_diagnose+dim_drugname+dim_quantity, 108)

    def forward(self, image, diagnose, drugname, quantity):
        # image
        # flatten image to 1d by view 
        image = image.view(-1, 107)
        image = self.fc_image(image)
        image = self.norm_image(image)
        image = self.act(image)
        # drop  
        image = self.dropout(image)



        # diagnose
        diagnose = self.fc_diagnose(diagnose)
        diagnose = self.norm_diagnose(diagnose)
        diagnose = self.act(diagnose)
        # drop
        diagnose = self.dropout(diagnose)
        diagnose = self.fc_diagnose_2(diagnose)
        diagnose = self.norm_diagnose_2(diagnose)
        diagnose = self.act(diagnose)

        # drugname
        drugname = self.fc_drugname(drugname)
        drugname = self.norm_drugname(drugname)
        drugname = self.act(drugname)
        # drop 
        drugname = self.dropout(drugname)
        drugname = self.fc_drugname_2(drugname)
        drugname = self.norm_drugname_2(drugname)
        drugname = self.act(drugname)

        # quantity
        quantity = self.norm_quantity(quantity)
        quantity = self.act(quantity)

        # concat
        concat = torch.cat((image, diagnose, drugname, quantity), dim=1)
        concat = self.norm_concat(concat)
        concat = self.dropout(concat)
        concat = self.fc_concat(concat)

        return concat
