import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(filepath):
    """Load and preprocess the car price dataset"""
    df = pd.read_csv(filepath)
    
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    
    # Extract numeric values from string columns
    df['max_power'] = df['max_power'].str.extract(r'(\d+\.?\d*)').astype(float)
    df['mileage'] = df['mileage'].str.split(expand=True)[0].astype(float)
    df['engine'] = df['engine'].str.split(expand=True)[0].astype(float)
    df['torque'] = df['torque'].str.split(' ', expand=True)[0].str.extract(r'(\d+\.?\d*)').astype(float)
    
    # Impute missing values using KNN
    imputer = KNNImputer()
    cols_to_impute = ['mileage', 'engine', 'max_power', 'torque', 'seats']
    imputed = imputer.fit_transform(df[cols_to_impute])
    
    for i, col in enumerate(cols_to_impute):
        df[col] = imputed[:, i]
    
    # Feature engineering: create age feature
    df['age'] = 2025 - df['year']
    df = df.drop(columns=['year'])
    
    # Reorder columns
    df = df[['name', 'age', 'km_driven', 'fuel', 'seller_type', 'transmission', 
             'owner', 'mileage', 'engine', 'max_power', 'torque', 'seats', 'selling_price']]
    
    return df

def create_pipeline():
    """Create the full preprocessing and modeling pipeline"""
    
    # Categorical columns encoding
    cat_cols = ['fuel', 'seller_type', 'transmission', 'owner']
    encoding = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    # Numerical columns transformation
    num_cols = ['age', 'km_driven', 'mileage', 'engine', 'max_power', 'torque', 'seats']
    num_handling = Pipeline([
        ('transformer', PowerTransformer(method='yeo-johnson')),
        ('scaling', StandardScaler())
    ])
    
    # Column transformer
    transformation = ColumnTransformer([
        ('cats', encoding, cat_cols),
        ('nums', num_handling, num_cols)
    ])
    
    # Full pipeline with polynomial features
    full_pipeline = Pipeline([
        ('pre', transformation),
        ('poly', PolynomialFeatures(include_bias=False, interaction_only=False)),
        ('reg', LinearRegression())
    ])
    
    return full_pipeline

def train_model(df):
    """Train the model using GridSearchCV"""
    
    # Prepare features and target
    X = df.drop(columns=['name', 'selling_price'])
    y = df['selling_price']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    
    # Create pipeline
    full_pipeline = create_pipeline()
    
    # Define parameter grid
    param_grid = [
        {
            'poly__degree': [1, 2],
            'reg': [LinearRegression()],
            'reg__fit_intercept': [True]
        },
        {
            'poly__degree': [1, 2, 3],
            'reg': [Ridge()],
            'reg__alpha': [0.01, 0.1, 1.0, 10.0],
            'reg__fit_intercept': [True]
        },
        {
            'poly__degree': [1, 2, 3],
            'reg': [ElasticNet(max_iter=10000)],
            'reg__alpha': [0.01, 0.1, 1.0, 10.0],
            'reg__fit_intercept': [True]
        }
    ]
    
    # GridSearchCV
    grid = GridSearchCV(
        estimator=full_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=2
    )
    
    print("Training model with GridSearchCV...")
    grid.fit(X_train, y_train)
    
    # Get best model
    best_model = grid.best_estimator_
    
    # Evaluate
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    print(f"\nBest Parameters: {grid.best_params_}")
    print(f"Train R²: {r2_train:.4f}")
    print(f"Test R²: {r2_test:.4f}")
    print(f"Test RMSE: {rmse_test:.2f}")
    
    return best_model, X_train, grid.best_params_

def save_model_and_metadata(model, X_train, best_params, df):
    """Save the trained model and metadata for deployment"""
    
    # Save the model
    joblib.dump(model, 'car_price_model.pkl')
    print("\nModel saved as 'car_price_model.pkl'")
    
    # Extract unique values for categorical features
    metadata = {
        'fuel_types': sorted(df['fuel'].unique().tolist()),
        'seller_types': sorted(df['seller_type'].unique().tolist()),
        'transmission_types': sorted(df['transmission'].unique().tolist()),
        'owner_types': sorted(df['owner'].unique().tolist()),
        'feature_ranges': {
            'age': {'min': int(df['age'].min()), 'max': int(df['age'].max()), 'mean': float(df['age'].mean())},
            'km_driven': {'min': int(df['km_driven'].min()), 'max': int(df['km_driven'].max()), 'mean': float(df['km_driven'].mean())},
            'mileage': {'min': float(df['mileage'].min()), 'max': float(df['mileage'].max()), 'mean': float(df['mileage'].mean())},
            'engine': {'min': float(df['engine'].min()), 'max': float(df['engine'].max()), 'mean': float(df['engine'].mean())},
            'max_power': {'min': float(df['max_power'].min()), 'max': float(df['max_power'].max()), 'mean': float(df['max_power'].mean())},
            'torque': {'min': float(df['torque'].min()), 'max': float(df['torque'].max()), 'mean': float(df['torque'].mean())},
            'seats': {'min': int(df['seats'].min()), 'max': int(df['seats'].max()), 'mean': float(df['seats'].mean())}
        },
        'best_params': best_params
    }
    
    # Save metadata
    joblib.dump(metadata, 'model_metadata.pkl')
    print("Metadata saved as 'model_metadata.pkl'")
    
    return metadata

if __name__ == "__main__":
    # Load and preprocess data
    print("Loading data...")
    df = load_and_preprocess_data('car_price.csv')
    
    # Train model
    model, X_train, best_params = train_model(df)
    
    # Save model and metadata
    metadata = save_model_and_metadata(model, X_train, best_params, df)
    
    print("\n✓ Model training and saving completed successfully!")
    print("Files created: car_price_model.pkl, model_metadata.pkl")