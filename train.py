# Author Haochen Li
'''This is the training file to train the model and do prune and quantization and save the model'''
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnx import version_converter

def load_data(train_file, test_file):
    '''Load data from csv and concatonate'''

    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    df = pd.concat([df_train, df_test])

    print("Read csv complete")
    print(df.info())
    print(df.head())

    return df # no NaN value

def preprocess(df):
    '''Clean data, one-hot encoding, split X&y, split train&test, standard scaler'''
    # drop column
    df.drop("Timestamp", axis = 1, inplace=True)

    # one hot encoding
    label_encoder = LabelEncoder()
    df['Class'] = label_encoder.fit_transform(df['Class'])

    # split into X and y
    X = df[["AccX","AccY","AccZ","GyroX","GyroY","GyroZ"]]
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # scaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, 'scaler.save')

    return X_train, X_test, y_train, y_test


def best_rf(X_train, X_test, y_train, y_test):
    '''Do grid search and find the best rf model'''
    print("finding best random forest...")
    # param grid
    # param_grid = {
    #     'n_estimators': [50, 100, 200],
    #     'max_depth': [3, 4, 5, None],
    #     'min_samples_split': [2, 3, 4, 5],
    #     'min_samples_leaf': [1, 2, 4],
    #     'bootstrap': [True, False]
    # }
    param_grid = {
        'n_estimators': [50],
        'max_depth': [4],
        'min_samples_split': [5],
        'min_samples_leaf': [2],
        'bootstrap': [True]
    }
    
    # start grid search
    rf = RandomForestClassifier(random_state=0)
    grid_search = GridSearchCV(
        estimator=rf, 
        param_grid=param_grid, 
        scoring="roc_auc_ovr",  #one vs rest
        cv=3, 
        verbose=1, # print information and result at each step
    ) 
    grid_search.fit(X_train, y_train)

    # best model and its parameters
    print("\nBest Model:")
    print(f"AUROC: {grid_search.best_score_:.4f}")
    print(f"Best Parameters: {grid_search.best_params_}")

    # retrain the model on X_train
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)
    y_pred_proba = best_model.predict_proba(X_test)
    y_test_pred = best_model.predict(X_test)
    test_auroc = roc_auc_score(y_test, y_pred_proba, multi_class="ovr")
    test_recall = recall_score(y_test, y_test_pred, average="macro")   # Macro-average recall
    test_precision = precision_score(y_test, y_test_pred, average="macro")  # Macro-average precision
    test_f1 = f1_score(y_test, y_test_pred, average="macro") 
    print("Retrained best model test AUROC: ", test_auroc)
    print("Retrained best model test recall: ", test_recall)
    print("Retrained best model test precision: ", test_precision)
    print("Retrained best model test F1 score: ", test_f1)

    return best_model, test_auroc

def quantize_model(best_rf_model, X_train):
    # save the model to onnx
    print("Exporting Random Forest model to ONNX format...")
    initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
    onnx_model_path = "core_model.onnx"
    onnx_model = convert_sklearn(best_rf_model, initial_types=initial_type)
    updated_model = version_converter.convert_version(onnx_model, 11) # quantization is not supported before ver11.
    
    with open(onnx_model_path, "wb") as f:
        f.write(updated_model.SerializeToString())
    print(f"ONNX model saved at {onnx_model_path}")

    # pply dynamic quantization using ONNX Runtime
    print("Applying dynamic quantization...")
    quantized_model_path = "core_model_INT8.onnx"
    quantize_dynamic(
        model_input=onnx_model_path,
        model_output=quantized_model_path,
        weight_type=QuantType.QInt8  # use 8-bit integer for weights
    )
    print(f"Quantized ONNX model saved at {quantized_model_path}")

if __name__ == "__main__":
    df = load_data(train_file=os.path.join("data","train_motion_data.csv"), test_file=os.path.join("data","test_motion_data.csv"))
    X_train, X_test, y_train, y_test = preprocess(df)

    # get the best rf model
    best_rf_model, best_rf_auroc = best_rf(X_train, X_test, y_train, y_test)

    # since torch does not support random forest efficiently, compressing is done by ONNX quantization
    quantize_model(best_rf_model, X_train)


