#!/bin/bash

# This downloads to the datasets/ folder
mkdir datasets

# Download ScienceDirect dataset (ASL-HG)
curl -L -o datasets/asl-hg.zip "https://data.mendeley.com/public-api/zip/j4y5w2c8w9/download/1"
echo "unzipping" ; unzip datasets/asl-hg.zip -d datasets
rm datasets/asl-hg.zip
mv datasets/ASL-HG\ American\ Sign\ Language\ Hand\ Gesture\ Image\ D/ASL_HG_36000/* datasets/asl-hg
echo "unzipping" ; unzip -q datasets/asl-hg/ASL_Processed_Images.zip -d datasets/asl-hg
rm datasets/asl-hg/ASL_Processed_Images.zip
echo "unzipping" ; unzip -q datasets/asl-hg/ASL_Raw_Images.zip -d datasets/asl-hg
rm datasets/asl-hg/ASL_Raw_Images.zip

# Download Kaggle synthetic asl dataset
curl -L -o datasets/synthetic-asl-dataset.zip "https://www.kaggle.com/api/v1/datasets/download/lexset/synthetic-asl-alphabet"
echo "unzipping" ; unzip -q datasets/synthetic-asl-dataset.zip -d datasets/synthetic-asl-dataset
rm datasets/synthetic-asl-dataset.zip
rm -rf datasets/synthetic-asl-dataset/Train_Alphabet/Blank
rm -rf datasets/synthetic-asl-dataset/Test_Alphabet/Blank
mv datasets/synthetic-asl-dataset/Train_Alphabet datasets/synthetic-asl-dataset/train
mv datasets/synthetic-asl-dataset/Test_Alphabet datasets/synthetic-asl-dataset/test

# Download Kaggle synthetic asl numbers
curl -L -o datasets/synthetic-asl-numbers.zip "https://www.kaggle.com/api/v1/datasets/download/lexset/synthetic-asl-numbers"
echo "unzipping" ; unzip -q datasets/synthetic-asl-numbers.zip -d datasets/synthetic-asl-numbers
rm datasets/synthetic-asl-numbers.zip
rm -rf datasets/synthetic-asl-numbers/Train_Nums/Blank
rm -rf datasets/synthetic-asl-numbers/Test_Nums/Blank

# Merge the synthetic asl alphabet and numbers
mv datasets/synthetic-asl-numbers/Train_Nums/* datasets/synthetic-asl-dataset/train
mv datasets/synthetic-asl-numbers/Test_Nums/* datasets/synthetic-asl-dataset/test
mv datasets/synthetic-asl-numbers/numbers.jpg datasets/synthetic-asl-dataset/numbers.jpg
rm -rf datasets/synthetic-asl-numbers

# Create missing folders
mkdir -p \
    datasets/asl-hg/asl_dataset/10 \
    datasets/asl-hg/asl_processed/test/10 \
    datasets/asl-hg/asl_processed/train/10 \
    datasets/synthetic-asl-dataset/test/0 \
    datasets/synthetic-asl-dataset/train/0

# du -sh datasets/* to see file size