# Backpack Price Prediction — Kaggle Playground Series S5E2

A deep learning regression project for Kaggle's **Backpack Prediction Challenge (Playground Series — Season 5, Episode 2)**. The goal is to predict backpack prices from a combination of categorical and numerical product attributes.

The project uses **TensorFlow/Keras** with a preprocessing pipeline that handles missing data, target encoding, feature standardisation, regularisation and early stopping.

## Project Overview

- **Task:** Tabular regression
- **Target:** `Price`
- **Training data:** `train.csv` combined with `training_extra.csv`
- **Combined training samples:** approximately 4.19 million
- **Evaluation metric:** Root Mean Squared Error (RMSE)
- **Environment:** Google Colab with GPU support
- **Competition:** https://www.kaggle.com/competitions/playground-series-s5e2

### Features

The model uses the following backpack attributes:

- Brand
- Material
- Size
- Compartments
- Laptop Compartment
- Waterproof
- Style
- Color
- Weight Capacity (kg)

The `id` column is excluded from model training.

## Preprocessing Pipeline

The notebook performs the following preprocessing steps:

1. **Data integration**  
   Combines the main training dataset with the additional training dataset.

2. **Missing-value handling**
   - Missing `Weight Capacity (kg)` values are replaced using the training-set median.
   - Missing categorical values are sampled according to the observed category distribution for each feature.

3. **Target encoding**  
   Features are transformed using RAPIDS cuML `TargetEncoder` with:
   - 25 folds
   - smoothing factor of 10
   - mean target statistic

4. **Feature standardisation**  
   `StandardScaler` is used to transform the encoded features to a common scale.

5. **Train-validation split**  
   The processed training data is split into:
   - 80% training
   - 20% validation  

   using `random_state=42`.

## Neural Network Architecture

The final regression model is implemented using TensorFlow/Keras:

```text
Input
  ↓
Dropout (0.1)
  ↓
Dense (128, L2 regularisation)
  ↓
Batch Normalisation
  ↓
ReLU
  ↓
Dropout (0.1)
  ↓
Dense (128, L2 regularisation)
  ↓
Batch Normalisation
  ↓
ReLU
  ↓
Dropout (0.1)
  ↓
Dense (64, L2 regularisation)
  ↓
Batch Normalisation
  ↓
ReLU
  ↓
Dropout (0.1)
  ↓
Dense (1, Linear)
```

The hidden layers use **L2 regularisation (`1e-4`)**, batch normalisation and dropout to reduce overfitting and improve training stability.

## Training

The final model is trained using:

- **Optimizer:** Adam
- **Loss:** Mean Squared Error (MSE)
- **Metric:** RMSE
- **Batch size:** 2048
- **Maximum epochs:** 30
- **Early stopping patience:** 5 epochs
- **Best-weight restoration:** Enabled
- **Data pipeline:** TensorFlow `tf.data` with prefetching

GPU detection and TensorFlow GPU memory-growth configuration are included for execution in Google Colab.

## Validation Performance

The notebook achieved a validation RMSE of approximately:

**38.68 RMSE**

Early stopping is used to restore the weights associated with the best validation loss.

## Hyperparameter Tuning Experiments

The notebook also contains experimental code for:

- K-fold cross-validation
- Optuna hyperparameter optimisation
- Learning-rate scheduling
- Optuna pruning

These sections are currently retained as experimental/commented code rather than being part of the final training pipeline.

## Technologies Used

- Python
- TensorFlow / Keras
- pandas
- NumPy
- scikit-learn
- RAPIDS cuML
- CuPy
- Optuna
- SciKeras
- Google Colab

## Running the Project

1. Open `BackPackPredictionLatest.ipynb` in Google Colab.

2. Enable a GPU runtime if available.

3. Place the required datasets in Google Drive:

```text
/content/drive/MyDrive/train.csv
/content/drive/MyDrive/training_extra.csv
/content/drive/MyDrive/test.csv
```

4. Run the notebook cells in order.

5. After training, predictions are generated for the test dataset and saved as:

```text
y_pred.csv
```

Before submitting the generated file to Kaggle, verify that its prediction column name and format match the competition's current `sample_submission.csv`.

## Repository

Main notebook:

[`BackPackPredictionLatest.ipynb`](./BackPackPredictionLatest.ipynb)

## Purpose

This project demonstrates an end-to-end deep learning workflow for large-scale tabular regression, including data preprocessing, categorical encoding, neural-network regularisation, GPU-based training and generation of test predictions.
