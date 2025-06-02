# Hand Gesture Recognition for Maze Navigation

## Project Overview

This project is the backend component of the final ML project, enabling navigation through a maze using hand gesture recognition. Users control movement in the maze by performing specific hand signs recognized by a machine learning model.

The frontend part of this project is provided [here](https://github.com/IshraqAhmedJamaluddin/MLOPs-Final-Project). This backend API handles model training, prediction, and serving functionalities.

---

## Project Scope

- Train machine learning models to classify hand gestures from landmark data.
- Log experiments and results using MLflow for better model management and reproducibility.
- Select the best performing model based on validation accuracy.
- Deploy a backend API that accepts hand gesture inputs and returns navigation commands.

---

## Repository Structure

- **research/**: Contains experiments, data exploration, and model training scripts with MLflow logging.
- **production/**: Contains deployment-ready code focused on serving the model via an API.

---

## Dataset

- Dataset used: Hand landmarks extracted from images, stored in `hand_landmarks_data.csv`.
- Contains 21 (x, y) coordinate points per sample and corresponding gesture labels.

---

## Data Exploration & Preprocessing

- Explored class distribution and feature statistics.
- Normalized landmarks by re-centering around the wrist and scaling by distance to the middle fingertip.
- Labels encoded with `LabelEncoder`.
- All preprocessing steps and visualizations logged in MLflow.

---

## Model Training and Selection

Multiple classifiers were trained and evaluated with MLflow experiment tracking:

| Model               | Accuracy         | Deployment Status   |
|---------------------|------------------|--------------------|
| **SVM**             | 0.9870 (±0.00006)| Production         |
| - Run 1             | 0.9870197300     |                    |
| - Run 2             | 0.9870289409     |                    |
| - Run 3             | 0.9871305484     |                    |
| - Run 4             | 0.9870197300     |                    |
| **XGBoost**         | 0.9849           | Staging            |
| **Random Forest**    | 0.9797           | Not deployed       |
| **Logistic Regression** | 0.9042        | Not deployed       |

> **Note:** The SVM model was chosen as the best model due to its highest accuracy of **98.7%** on the validation set.

---

## MLflow Usage

- MLflow logs dataset parameters, metrics, model artifacts, and plots.
- Experiments are under the `Hand_Gesture_Recognition` experiment.
- Artifacts include class distribution charts, feature distributions, sample hand landmark visualizations, and trained model files.

---

## How to Use

### 1. Clone the repository:

bash
git clone <your_repo_url>
cd research

### 2. Setup environment:
pip install -r requirements.txt

###3. Run training script:
bash
python train.py

This runs preprocessing, trains multiple models, logs results in MLflow, and saves the best model (svm_model.pkl).

4. Use the trained model in backend API:
Load the saved SVM model for inference to classify hand gestures and control maze navigation.
