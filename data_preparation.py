import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

def prepare_data():
    # Load data
    df = pd.read_csv('data/train.csv')
    
    # Extract columns representing wilderness areas and soil types
    wilderness_columns = [col for col in df.columns if 'Wilderness_Area' in col]
    soil_columns = [col for col in df.columns if 'Soil_Type' in col]
    
    # Combine the wilderness and soil types for categorical features
    cat_features = wilderness_columns + soil_columns
    
    # Select numeric features
    numeric_features = df.drop(cat_features + ['Cover_Type', 'Id'], axis=1)
    
    # Combine numeric and categorical features
    X = pd.concat([numeric_features, df[cat_features]], axis=1)
    y = df['Cover_Type']
    
    # Generate interaction terms
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(X)
    X_poly_df = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(X.columns))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_poly_df, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, poly

# Data augmentation function
def augment_data(X, y, classes_to_augment=[1, 2], augmentation_factor=2):
    aug_X = X[y.isin(classes_to_augment)]
    aug_y = y[y.isin(classes_to_augment)]
    
    aug_X = pd.concat([aug_X] * augmentation_factor, ignore_index=True)
    aug_y = pd.concat([aug_y] * augmentation_factor, ignore_index=True)
    
    X_augmented = pd.concat([X, aug_X], ignore_index=True)
    y_augmented = pd.concat([y, aug_y], ignore_index=True)
    
    return X_augmented, y_augmented

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, poly = prepare_data()
    X_train_aug, y_train_aug = augment_data(X_train, y_train)
    print("Original training set shape:", X_train.shape)
    print("Augmented training set shape:", X_train_aug.shape)

    