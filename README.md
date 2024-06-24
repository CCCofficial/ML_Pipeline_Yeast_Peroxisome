
# ML Pipeline for Yeast Peroxisome Prediction

This repository contains the code and data for building machine learning models to predict yeast peroxisome capacity.

## Project Overview
This project aims to develop and evaluate various machine learning models for predicting the capacity of yeast peroxisomes. It includes data preprocessing, model training, and evaluation scripts.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/CCCofficial/ML_Pipeline_Yeast_Peroxisome.git
   ```
2. Navigate to the project directory:
   ```sh
   cd ML_Pipeline_Yeast_Peroxisome
   ```
3. Run the Jupyter notebooks provided to train and evaluate the models

## Usage
1. Run the Jupyter notebooks provided to train and evaluate the models:
   - `CNN_LSTM_02062023_117samples_first_round.ipynb`
     - Train and evaluate a CNN and CNN+LSTM model for the first round data (117 samples)
   - `CNN_LSTM_GBR_05152023_200samples_final_round.ipynb`
     - Train and evaluate a GBR and CNN+LSTM model for the final round data (200 samples)
   - `Compare_five_ML_models_117samples_first_round.ipynb`
     - Compare the performance of five different machine learning models on the first round data
   - `GBR_02062023_117samples_first_round.ipynb`
     - Train and evaluate a GBR model on the first round data
   - `Predictions_dim_reduction.ipynb`
     - Perform tsne, UMAP to the top predictions from GBR, CNN, and CNN+LSTM models
   - `Screening_prediction_cnn+lstm_05152023_ensemble_final_round.ipynb`
     - Ensembel screening predictions from CNN+LSTM on the final round
   - `Screening_prediction_gbr_05152023_final_round.ipynb`
     - Screening predictions from GBR on the final round
   - `gene_combin_generator.ipynb`
     - Apply novel combination selection algorithm where we randomly generated samples with gene combinations from the under-represesnted region. 
     

## Features
- **Data preprocessing and cleaning**: Scripts and methods to preprocess and clean the dataset.
- **Model training**: Training scripts for different machine learning models including CNN, LSTM, and GBR.
- **Model evaluation and comparison**: Evaluation metrics and comparison of different models.
- **Dimensionality reduction**: Techniques for reducing the dimensionality of the dataset.
- **Predictions**: Scripts for making predictions using the trained models.

## Contributing
Contributions are welcome! Please fork this repository and submit pull requests with your improvements.

## License
This project is licensed under the MIT License.

## Contact
For any questions or issues, please open an issue in this repository or contact:
- Jie Shi: shijie@ibm.com
- Shangying Wang: swang@altoslabs.com
- Sara Capponi: sara.capponi@ibm.com