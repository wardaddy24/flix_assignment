import pandas as pd
import os
from PIL import Image
import glob

# from sklearn.preprocessing import OneHotEncoder
# enc = OneHotEncoder(handle_unknown='ignore')
# df_attributes = df_attributes[df_attributes.columns[1:4]]
# df_attributes = enc.fit_transform(df_attributes)
# print(df_attributes.toarray())
# print(enc.get_feature_names())
# print(df_attributes.head())
# array = enc.get_feature_names(['neck','sleeve_length','pattern'])
# print(len(array))
# print(enc)

df = pd.read_csv('attributes.csv')
df =  df.dropna()

files = glob.glob('images/*')
# print(len(files))
count = 0
def image_exists(filename):
    filepath = "images\\"+filename
    return os.path.isfile(filepath)

# for i in df_attributes['filename']:
#     name = "images\\"+i
#     if name not in files:
#         print(os.path.isfile(name))
#         count+=1
# print(count)
print("Total:",df.count())
df_with_images = df[df["filename"].apply(image_exists)]
print("Valid Images:",df_with_images.count())


# one hot encode the Neck attribute
one_hot_neck = pd.get_dummies( df_with_images.neck, prefix='neck')

# one hot encode the sleeve_length attribute
one_hot_sleeve_length = pd.get_dummies( df_with_images.sleeve_length, prefix='sleeve_length')

# one hot encode the patter attribute
one_hot_pattern = pd.get_dummies( df_with_images.pattern, prefix='pattern')

# concatenate the one hot encoded attributes to dataframe
df_with_images = pd.concat([df_with_images, one_hot_neck, one_hot_sleeve_length, one_hot_pattern], axis=1)
