# Importing Necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
import pickle
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient
import os
from datetime import datetime
from sklearn.model_selection import ParameterGrid


# Initialize MLflow
mlflow.set_tracking_uri("./mlruns")  
mlflow.set_experiment("Hand_Gesture_Recognition")

print("MLflow tracking initialized!")
print(f"Tracking URI: {mlflow.get_tracking_uri()}")
print(f"Experiment: {mlflow.get_experiment_by_name('Hand_Gesture_Recognition')}")

# **Loading and Exploring The Dataset**
raw_data = pd.read_csv("D:/Hand_Gesture/Hand-Gesture-Research/data/hand_landmarks_data.csv")

# Start MLflow run for data exploration
with mlflow.start_run(run_name="Data_Exploration") as parent_run:
    # Log dataset info
    mlflow.log_param("dataset_shape", raw_data.shape)
    mlflow.log_param("num_features", raw_data.shape[1] - 1)  
    mlflow.log_param("num_samples", raw_data.shape[0])
    
    # Examining the shape of the dataset
    print(f"Dataset shape: {raw_data.shape}")
    
    # Examining the dataset structure
    print("\nDataset info:")
    raw_data.info()
    
    # Examining the columns in the dataset
    print("\nColumns (features of the dataset):")
    print(raw_data.columns.tolist())
    
    # Displaying summary statistics
    print("\nSummary statistics:")
    print(raw_data.describe())
    
    # Examining the labels
    print("\nUnique labels:")
    unique_labels = raw_data["label"].unique()
    print(unique_labels)
    mlflow.log_param("unique_labels", unique_labels.tolist())
    mlflow.log_param("num_classes", len(unique_labels))
    
    # Class distribution
    class_counts = raw_data["label"].value_counts()
    print("\nClass distribution:")
    print(class_counts)
    
    # Log class distribution
    for label, count in class_counts.items():
        mlflow.log_metric(f"class_count_{label}", count)
    
    # Visualizing class distribution
    plt.figure(figsize=(19, 9))
    sns.barplot(x=class_counts.index, y=class_counts.values)
    plt.xticks(rotation=45)
    plt.xlabel("Gesture Class")
    plt.ylabel("Count")
    plt.title("Class Distribution of Hand Gestures")
    plt.tight_layout()
    plt.savefig("class_distribution.png")
    mlflow.log_artifact("class_distribution.png")
    plt.show()
    
    # **Data Quality Checks**
    null_count = raw_data.isna().sum().sum()
    duplicate_count = raw_data.duplicated().sum()
    
    print(f"\nNull values: {null_count}")
    print(f"Duplicate rows: {duplicate_count}")
    
    mlflow.log_metric("null_values", null_count)
    mlflow.log_metric("duplicate_rows", duplicate_count)

# **Feature Distribution Visualization**
def plot_feature_distributions(data, columns_per_row=5):
    """Plots the distribution of numeric features in the given dataset."""
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    num_rows = (len(numeric_columns) // columns_per_row) + (len(numeric_columns) % columns_per_row > 0)
    
    plt.figure(figsize=(15, num_rows * 4))
    
    for i, column in enumerate(numeric_columns):
        plt.subplot(num_rows, columns_per_row, i + 1)
        sns.histplot(data[column], kde=True)
        plt.title(f'Distribution of {column}')
    
    plt.tight_layout()
    plt.savefig("feature_distributions.png")
    plt.show()
    return "feature_distributions.png"

#Hand Landmarks Visualization
def plot_hand_landmarks(df, sample_index=0):
    """Visualizes the hand landmarks (x, y) for a given sample."""
    sample = df.iloc[sample_index]
    
    x_points = [sample[f'x{i}'] for i in range(1, 22)]
    y_points = [sample[f'y{i}'] for i in range(1, 22)]
    
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
        (5, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
        (9, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
        (0, 17), (13, 17), (17, 18), (18, 19), (19, 20)  # Pinky finger
    ]
    
    plt.figure(figsize=(6, 6))
    plt.scatter(x_points, y_points, color='red', label="Keypoints")
    
    for connection in connections:
        x1, y1 = x_points[connection[0]], y_points[connection[0]]
        x2, y2 = x_points[connection[1]], y_points[connection[1]]
        plt.plot([x1, x2], [y1, y2], 'b', linewidth=2)
    
    gesture_label = sample.get('label', 'Unknown Gesture')
    plt.title(f"Hand Landmarks Visualization (Sample {sample_index})\nGesture: {gesture_label}")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.legend()
    
    filename = f"hand_landmarks_sample_{sample_index}_{gesture_label}.png"
    plt.savefig(filename)
    plt.show()
    return filename

# Data Preprocessing with MLflow Tracking
with mlflow.start_run(run_name="Data_Preprocessing") as preprocess_run:
    # Plot feature distributions
    feature_dist_file = plot_feature_distributions(raw_data)
    mlflow.log_artifact(feature_dist_file)
    
    # Plot hand landmarks for each class
    unique_classes = raw_data['label'].unique()
    landmark_files = []
    
    for class_label in unique_classes:
        sample_index = raw_data[raw_data['label'] == class_label].index[0]
        landmark_file = plot_hand_landmarks(raw_data, sample_index)
        landmark_files.append(landmark_file)
        mlflow.log_artifact(landmark_file)
    
    # Label Mapping
    label_mapping = {label: idx for idx, label in enumerate(raw_data["label"].unique())}
    inverse_mapping = {v: k for k, v in label_mapping.items()}
    
    mlflow.log_dict(label_mapping, "label_mapping.json")
    mlflow.log_dict(inverse_mapping, "inverse_mapping.json")
    
    # Label Encoding
    encoder = LabelEncoder()
    raw_data["label"] = encoder.fit_transform(raw_data["label"])
    
    # Save encoder
    with open('encoder.pkl', 'wb') as file:
        pickle.dump(encoder, file)
    mlflow.log_artifact('encoder.pkl')
    
    # Normalization of Hand Landmarks
    def normalize_landmarks_df(raw_data: pd.DataFrame, verbose=False) -> pd.DataFrame:
        """Normalize hand landmarks by re-centering around wrist and scaling by middle fingertip distance."""
        raw_data = raw_data.copy()
        
        wrist_x = raw_data['x1'].values
        wrist_y = raw_data['y1'].values
        middle_tip_x = raw_data['x13'].values
        middle_tip_y = raw_data['y13'].values
        
        scale_factor = np.sqrt((middle_tip_x - wrist_x)**2 + (middle_tip_y - wrist_y)**2)
        
        for i in range(1, 22):
            raw_data[f'x{i}'] = (raw_data[f'x{i}'] - wrist_x) / scale_factor
            raw_data[f'y{i}'] = (raw_data[f'y{i}'] - wrist_y) / scale_factor
        
        if verbose:
            wrist_mean = raw_data[['x1', 'y1']].mean().values
            middle_dist = np.sqrt(raw_data['x13']**2 + raw_data['y13']**2).mean()
            print("Wrist (x1, y1) after normalization:", wrist_mean)
            print("Distance to middle fingertip (should be ~1.0):", middle_dist)
            mlflow.log_metric("wrist_x_mean_after_norm", wrist_mean[0])
            mlflow.log_metric("wrist_y_mean_after_norm", wrist_mean[1])
            mlflow.log_metric("middle_fingertip_distance_after_norm", middle_dist)
        
        return raw_data
    
    normalized_df = normalize_landmarks_df(raw_data, verbose=True)
    
    # Save preprocessed data
    normalized_df.to_csv('./preprocessed_df.csv', index=None)
    mlflow.log_artifact('./preprocessed_df.csv')
    
    mlflow.log_param("normalization_method", "wrist_centered_middle_finger_scaled")

# **Data Splitting with MLflow Tracking**
with mlflow.start_run(run_name="Data_Splitting") as split_run:
    data = pd.read_csv('./preprocessed_df.csv')
    
    # Features-labels split
    features = data.drop(["label"], axis=1)
    labels = data["label"]
    
    # Train-validation-test split
    features_train, features_validation_test, labels_train, labels_validation_test = train_test_split(
        features, labels, test_size=0.3, random_state=100)
    features_validation, features_test, labels_validation, labels_test = train_test_split(
        features_validation_test, labels_validation_test, test_size=0.5, random_state=100)
    
    # Log split information
    mlflow.log_param("train_size", len(features_train))
    mlflow.log_param("validation_size", len(features_validation))
    mlflow.log_param("test_size", len(features_test))
    mlflow.log_param("test_split_ratio", 0.3)
    mlflow.log_param("validation_split_ratio", 0.15)
    mlflow.log_param("random_state", 100)
    
    print(f"Training set size: {len(features_train)}")
    print(f"Validation set size: {len(features_validation)}")
    print(f"Test set size: {len(features_test)}")

# **Model Training with MLflow Tracking**

def evaluate_model(model, features_val, labels_val, model_name):
    """Evaluate model and return metrics."""
    predictions = model.predict(features_val)
    accuracy = accuracy_score(labels_val, predictions)
    precision = precision_score(labels_val, predictions, average='weighted')
    recall = recall_score(labels_val, predictions, average='weighted')
    f1 = f1_score(labels_val, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': predictions
    }

def log_model_metrics(metrics, model_name):
    """Log model metrics to MLflow."""
    mlflow.log_metric("accuracy", metrics['accuracy'])
    mlflow.log_metric("precision", metrics['precision'])
    mlflow.log_metric("recall", metrics['recall'])
    mlflow.log_metric("f1_score", metrics['f1_score'])

# **Baseline Models Training**
with mlflow.start_run(run_name="Baseline_Models_Comparison") as baseline_run:
    models = {
        "Logistic_Regression": LogisticRegression(),
        "Decision_Tree": DecisionTreeClassifier(),
        "SVM": SVC(),
        "Random_Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier()
    }
    
    baseline_results = []
    
    for name, model in models.items():
        with mlflow.start_run(run_name=f"Baseline_{name}", nested=True):
            # Train model
            model.fit(features_train, labels_train)
            
            # Evaluate model
            metrics = evaluate_model(model, features_validation, labels_validation, name)
            
            # Log parameters (default parameters)
            mlflow.log_params(model.get_params())
            
            # Log metrics
            log_model_metrics(metrics, name)
            
            # Log model
            if name == "XGBoost":
                mlflow.xgboost.log_model(model, f"baseline_{name.lower()}_model")
            else:
                mlflow.sklearn.log_model(model, f"baseline_{name.lower()}_model")
            
            baseline_results.append([name, metrics['accuracy'], metrics['precision'], 
                                   metrics['recall'], metrics['f1_score']])
            
            print(f"{name} - Accuracy: {metrics['accuracy']:.4f}")
    
    # Create and log baseline results DataFrame
    baseline_df = pd.DataFrame(baseline_results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"])
    baseline_df.to_csv("baseline_results.csv", index=False)
    mlflow.log_artifact("baseline_results.csv")
    
    print("\nBaseline Results:")
    print(baseline_df)

# **Hyperparameter Tuning with MLflow Tracking**

# Custom GridSearchCV with MLflow logging
def mlflow_grid_search(estimator, param_grid, X_train, y_train, X_val, y_val, 
                      model_name, cv=5):
    """Custom GridSearchCV with MLflow logging for each trial."""
    
    best_score = -1
    best_params = None
    best_estimator = None
    
    # Generate all parameter combinations
    from sklearn.model_selection import ParameterGrid
    param_list = list(ParameterGrid(param_grid))
    
    for i, params in enumerate(param_list):
        with mlflow.start_run(run_name=f"Trial_{i+1}", nested=True):
            # Set parameters and train model
            model = estimator.__class__(**params)
            model.fit(X_train, y_train)
            
            # Evaluate model
            metrics = evaluate_model(model, X_val, y_val, f"{model_name}_trial_{i+1}")
            
            # Log parameters and metrics
            mlflow.log_params(params)
            log_model_metrics(metrics, f"{model_name}_trial_{i+1}")
            
            # Check if this is the best model so far
            if metrics['accuracy'] > best_score:
                best_score = metrics['accuracy']
                best_params = params
                best_estimator = model
            
            print(f"Trial {i+1}/{len(param_list)} - Accuracy: {metrics['accuracy']:.4f}")
    
    return best_estimator, best_params, best_score

# **SVM Hyperparameter Tuning**
with mlflow.start_run(run_name="SVM_Hyperparameter_Tuning") as svm_parent_run:
    svm_parameters = {
        'kernel': ['rbf'],
        'C': [0.01, 0.1, 1, 10, 100],
        'gamma': [0.01, 0.1, 1, 10, 100]
    }
    
    mlflow.log_params({
        "model_type": "SVM",
        "tuning_method": "GridSearchCV",
        "cv_folds": 5,
        "param_combinations": len(list(ParameterGrid(svm_parameters)))
    })
    
    svm_winner, best_svm_params, best_svm_score = mlflow_grid_search(
        SVC(), svm_parameters, features_train, labels_train, 
        features_validation, labels_validation, "SVM"
    )
    
    # Log best results to parent run
    mlflow.log_params(best_svm_params)
    mlflow.log_metric("best_accuracy", best_svm_score)
    mlflow.sklearn.log_model(svm_winner, "best_svm_model")
    
    print(f"\nBest SVM Parameters: {best_svm_params}")
    print(f"Best SVM Accuracy: {best_svm_score:.4f}")

# **Logistic Regression Hyperparameter Tuning**
with mlflow.start_run(run_name="LogisticRegression_Hyperparameter_Tuning") as lr_parent_run:
    logreg_parameters = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs'],
        'max_iter': [100, 200, 300]
    }
    
    mlflow.log_params({
        "model_type": "LogisticRegression",
        "tuning_method": "GridSearchCV",
        "cv_folds": 5,
        "param_combinations": len(list(ParameterGrid(logreg_parameters)))
    })
    
    logreg_winner, best_lr_params, best_lr_score = mlflow_grid_search(
        LogisticRegression(), logreg_parameters, features_train, labels_train,
        features_validation, labels_validation, "LogisticRegression"
    )
    
    # Log best results to parent run
    mlflow.log_params(best_lr_params)
    mlflow.log_metric("best_accuracy", best_lr_score)
    mlflow.sklearn.log_model(logreg_winner, "best_logreg_model")
    
    print(f"\nBest Logistic Regression Parameters: {best_lr_params}")
    print(f"Best Logistic Regression Accuracy: {best_lr_score:.4f}")

# **Random Forest Hyperparameter Tuning**
with mlflow.start_run(run_name="RandomForest_Hyperparameter_Tuning") as rf_parent_run:
    rf_parameters = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    
    mlflow.log_params({
        "model_type": "RandomForest",
        "tuning_method": "GridSearchCV",
        "cv_folds": 5,
        "param_combinations": len(list(ParameterGrid(rf_parameters)))
    })
    
    rf_winner, best_rf_params, best_rf_score = mlflow_grid_search(
        RandomForestClassifier(), rf_parameters, features_train, labels_train,
        features_validation, labels_validation, "RandomForest"
    )
    
    # Log best results to parent run
    mlflow.log_params(best_rf_params)
    mlflow.log_metric("best_accuracy", best_rf_score)
    mlflow.sklearn.log_model(rf_winner, "best_rf_model")
    
    print(f"\nBest Random Forest Parameters: {best_rf_params}")
    print(f"Best Random Forest Accuracy: {best_rf_score:.4f}")

# **XGBoost Hyperparameter Tuning**
with mlflow.start_run(run_name="XGBoost_Hyperparameter_Tuning") as xgb_parent_run:
    xgb_parameters = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    
    mlflow.log_params({
        "model_type": "XGBoost",
        "tuning_method": "GridSearchCV",
        "cv_folds": 5,
        "param_combinations": len(list(ParameterGrid(xgb_parameters)))
    })
    
    xgb_winner, best_xgb_params, best_xgb_score = mlflow_grid_search(
        XGBClassifier(), xgb_parameters, features_train, labels_train,
        features_validation, labels_validation, "XGBoost"
    )
    
    # Log best results to parent run
    mlflow.log_params(best_xgb_params)
    mlflow.log_metric("best_accuracy", best_xgb_score)
    mlflow.xgboost.log_model(xgb_winner, "best_xgb_model")
    
    print(f"\nBest XGBoost Parameters: {best_xgb_params}")
    print(f"Best XGBoost Accuracy: {best_xgb_score:.4f}")

# **Final Model Comparison and Selection**
with mlflow.start_run(run_name="Final_Model_Comparison") as final_run:
    # Evaluate all tuned models
    tuned_models = {
        "Tuned_SVM": svm_winner,
        "Tuned_LogisticRegression": logreg_winner,
        "Tuned_RandomForest": rf_winner,
        "Tuned_XGBoost": xgb_winner
    }
    
    final_results = []
    best_overall_score = -1
    best_overall_model = None
    best_overall_name = None
    
    for name, model in tuned_models.items():
        with mlflow.start_run(run_name=f"Final_{name}", nested=True):
            metrics = evaluate_model(model, features_validation, labels_validation, name)
            
            # Log final metrics
            log_model_metrics(metrics, name)
            
            # Log model
            if "XGBoost" in name:
                mlflow.xgboost.log_model(model, f"final_{name.lower()}_model")
            else:
                mlflow.sklearn.log_model(model, f"final_{name.lower()}_model")
            
            final_results.append([name, metrics['accuracy'], metrics['precision'], 
                                metrics['recall'], metrics['f1_score']])
            
            # Track best overall model
            if metrics['accuracy'] > best_overall_score:
                best_overall_score = metrics['accuracy']
                best_overall_model = model
                best_overall_name = name
    
    # Create and log final results
    final_df = pd.DataFrame(final_results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"])
    final_df.to_csv("final_model_comparison.csv", index=False)
    mlflow.log_artifact("final_model_comparison.csv")
    
    # Log best overall model information
    mlflow.log_param("best_overall_model", best_overall_name)
    mlflow.log_metric("best_overall_accuracy", best_overall_score)
    
    print("\nFinal Model Comparison:")
    print(final_df)
    print(f"\nBest Overall Model: {best_overall_name} with accuracy: {best_overall_score:.4f}")

# **Test Set Evaluation of Best Model**
with mlflow.start_run(run_name="Best_Model_Test_Evaluation") as test_run:
    # Test the best model on the test set
    test_metrics = evaluate_model(best_overall_model, features_test, labels_test, "best_model_test")
    
    # Log test metrics
    mlflow.log_param("model_name", best_overall_name)
    mlflow.log_metric("test_accuracy", test_metrics['accuracy'])
    mlflow.log_metric("test_precision", test_metrics['precision'])
    mlflow.log_metric("test_recall", test_metrics['recall'])
    mlflow.log_metric("test_f1_score", test_metrics['f1_score'])
    
    # Generate and log confusion matrix
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(labels_test, test_metrics['predictions'])
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Test Set')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix_test.png')
    mlflow.log_artifact('confusion_matrix_test.png')
    plt.show()
    
    # Generate and log classification report
    report = classification_report(labels_test, test_metrics['predictions'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv('classification_report_test.csv')
    mlflow.log_artifact('classification_report_test.csv')
    
    # Save the best model
    if "XGBoost" in best_overall_name:
        mlflow.xgboost.log_model(best_overall_model, "production_model")
    else:
        mlflow.sklearn.log_model(best_overall_model, "production_model")
    
    # Also save using joblib for compatibility
    joblib.dump(best_overall_model, 'best_model_production.pkl')
    mlflow.log_artifact('best_model_production.pkl')
    
    print(f"\nTest Set Results for {best_overall_name}:")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")
    print(f"Test F1-Score: {test_metrics['f1_score']:.4f}")

# **Video Processing Integration** (Updated with MLflow model loading)
print("\n" + "="*50)
print("MLflow Integration Complete!")
print("="*50)
print(f"Best Model: {best_overall_name}")
print(f"Best Validation Accuracy: {best_overall_score:.4f}")
print(f"Final Test Accuracy: {test_metrics['accuracy']:.4f}")
print("\nAll experiments have been tracked in MLflow.")
print("Run 'mlflow ui' in your terminal to view the experiment dashboard.")

