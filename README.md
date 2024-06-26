
# ML Pipeline for Yeast Peroxisome Prediction


## Project Overview
This project aims to develop and evaluate various machine learning models for predicting the capacity of yeast peroxisomes. It includes data preprocessing, model training, evaluation, and _in silico_ screening, among other tasks. More details can be found below under __Usage__.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)
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
     

## Contributing
Contributions are welcome! Please fork this repository and submit pull requests with your improvements.

## Acknowledgments
This work is funded by the National Science Foundation (NSF) grant No. DBI-1548297, Center for Cellular Construction.

Disclaimer: Any opinions, findings and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation.

## License
This project is licensed under the MIT License.

## Contact
For any questions or issues, please open an issue in this repository or contact:
- Jie Shi: shijie@ibm.com
- Shangying Wang: swang@altoslabs.com
- Sara Capponi: sara.capponi@ibm.com

