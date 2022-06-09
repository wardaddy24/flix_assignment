# flix_assignment

## Data Preprocessing:
1. Dropped all NA rows.
2. Dropping all rows for which images are not present in the image folder.
3. One hot encoding.

## Model Training - python train.py
1. Epochs = 30
2. Using pretrained resnet 50  
3. Total output = 21 (got to know from one-hot )
4. loss.png for training and validation loss is present

## Inference - python inference.py
1. Uses model.pth to do inference
2. saves files in the outputs folder with predicted and actual labels on it.
 
