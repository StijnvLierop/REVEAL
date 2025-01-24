import os

import pandas as pd
from tqdm import tqdm
from creation.utils import bpnzac


df = pd.read_csv('/mnt/extern/EU Restricted/Backup_Harde_Schijf/picture_dataset_latest/dataset_master_17_01_2025.csv', sep=';', decimal=',')

# def transform_er(er):
#     if er == "1,00E-05":
#         er = "0.00001"
#     return er
#
# df['embeddingRate'] = df['embeddingRate'].apply(transform_er).astype(float)

# print(df.groupby('tool')['embeddingRate'].max())

tqdm.pandas()

df['bpnzAC'] = 0
df['bpnzAC'] = df[df['embeddingRate'] != 0].progress_apply(lambda row: bpnzac(os.path.join('/mnt/extern/EU Restricted/dataset_pictures_finalfinal/Public_Set_Stego_Pictures',row['stegoPictureName']),
                                                                     os.path.join('/mnt/extern/EU Restricted/Backup_Harde_Schijf/picture dataset/code/messages',row['message'])), axis=1)

# print(bpnzac('/mnt/extern/EU Restricted/dataset_pictures_finalfinal/Public_Set_Stego_Pictures/11.jpeg',
#              ))


df.to_csv('/mnt/extern/EU Restricted/Backup_Harde_Schijf/picture_dataset_latest/dataset_master_17_01_2025_with_bpnzac.csv', index=False, sep=';', decimal=',')