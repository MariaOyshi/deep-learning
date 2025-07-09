# Brain Tumor Segmentation 

This project uses deep learning to automatically segment brain tumors from MRI images.

##  Dataset
- Source: BRATS Dataset (multi-modal MRI)

##  Preprocessing
- Z-score normalization
- Binarization (FLAIR, T1ce)
- Morphological filtering

##  Model
- CNN / U-Net with attention mechanism
- Achieves high Dice Score on BRATS 2020

## Deployment
- Streamlit/Kaggle-based interface
- Upload MRI image → Get tumor segmentation

## Demo
 [Kaggle Notebook Link](https://www.kaggle.com/code/mariaoyshi/brain-tumor-segmentation)

