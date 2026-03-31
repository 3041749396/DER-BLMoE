# DER-BLMoE
This repository contains the source code for the paper "Robust and Efficient Specific Emitter Identification via Broad Learning-Based Mixture of Experts" for your reference.
## 🚀 Usage Guidelines

1. **Download the Dataset:** The original ADS-B dataset can be downloaded from this link: `https://gitee.com/heu-linyun/Codes`

2. **Data Preprocessing:** Please place the original ADS-B dataset into the `source` folder, and then run the `process.mlx` file in MATLAB to perform initial preprocessing. This step includes signal smoothing, complex baseband signal construction, DC removal (de-biasing), amplitude normalization, and splitting/concatenation. The preprocessed ADS-B dataset will be automatically saved in the `Datasets` folder.

3. **Configuration:** You can navigate to the `configs/config.py`to customize various hyperparameters, such as the dataset name, number of classes, number of experts, and number of feature windows. Please adjust these parameters according to your specific needs before running the code. 
   > **Note:** The default parameter settings in this repository are identical to those used in the original paper. If your goal is to reproduce the results presented in the paper, we recommend keeping the default settings unchanged.

4. **Running the Experiments:** * First, run `experiments/exp_train.py` to train the model and generate the corresponding model files. 
   * Once the training is complete, you can run `experiments/exp_test.py` and other related experimental scripts to evaluate the performance.

5. **Contact:** If you have any questions or encounter any issues, please feel free to contact the author, Xiao Yan, via email at: [1225014125@njupt.edu.cn](mailto:1225014125@njupt.edu.cn).

6. **Future Updates:** Please note that a cleaner, more polished, and highly readable version of the code will be released upon the acceptance of our paper.
