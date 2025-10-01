
Place your MRI data into the data/ directory with the following structure for clear training, validation, and testing splits:


data/
  train/
    T1/
    T2/
    FLAIR/
  val/
    T1/
    T2/
    FLAIR/
  test/
    T1/
    T2/
    FLAIR/
Each subfolder (T1, T2, FLAIR) contains the corresponding modality images for that split.

Dataset Overview
The dataset comprises 7500 preprocessed MRI samples from the BraTS 2020 collection, split into training (70%), validation (15%), and testing (15%). This split provides a balanced, high-quality foundation for robust and generalizable brain tumor detection through multimodal MRI feature fusion.

