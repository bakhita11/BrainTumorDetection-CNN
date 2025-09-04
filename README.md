CNN-Based Brain Tumor Detection using Multimodal Image Fusion
This repository contains code and data for convolutional neural network (CNN) based brain tumor detection using fused MRI, PET, and CT images.
## Installation
Install dependencies with:                    pip install -r requirements.txt
                                                                                                                 
## Usage
- Place the dataset in the `data/` folder.
- Run training scripts from `scripts/` folder.
- Generated results and models are saved to `results/`.
## Citation
Please cite our paper if you use this code.



Dataset folder: Tumor_Type folder with subfolders brain_glioma, brain_menin, and brain_tumor.
Dataset is organized in the following structure for training, validation, and testing:

Tumor_Type/
    train/
        brain_glioma/
        brain_menin/
        brain_tumor/
    val/
        brain_glioma/
        brain_menin/
        brain_tumor/
    test/
        brain_glioma/
        brain_menin/
        brain_tumor/

