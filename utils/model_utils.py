import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def train_model(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LinearRegression().fit(X_scaled, y)
    return model, scaler

def save_model_and_scaler(model, scaler, project_root):
    model_dir = os.path.join(project_root, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(model, os.path.join(model_dir, 'linear_regression_model.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
