import streamlit as st
import pandas as pd
import numpy as np
import sklearn
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
import shap

encoder = OneHotEncoder(sparse_output=False)

st.write(f"Current scikit-learn version: {sklearn.__version__}")
st.write(f"NumPy version: {np.__version__}")
st.write(f"SHAP version: {shap.__version__}")

# 加载保存的版本信息
try:
    with open('sklearn_version.txt', 'r') as f:
        saved_version = f.read().strip()
    st.write(f"Model trained with scikit-learn version: {saved_version}")
except FileNotFoundError:
    st.write("Unable to find saved scikit-learn version information.")


# Load models and preprocessing objects
models = {
    'RandomForest': joblib.load('RandomForest_model_compressed.joblib'),
    'ExtraTrees': joblib.load('ExtraTrees_model_compressed.joblib'),
    'LightGBM': joblib.load('LightGBM_model_compressed.joblib'),
    'CatBoost': joblib.load('CatBoost_model_compressed.joblib'),
    'VotingClassifier': joblib.load('VotingClassifier_model_compressed.joblib')
}
poly = joblib.load('poly_compressed.joblib')

# Load a sample of the dataset for visualization
df = pd.read_csv('data/train.csv')

st.title('Forest Cover Type Classifier')

st.write("""
This app predicts the forest cover type based on cartographic variables.
It uses an ensemble of machine learning models and provides detailed visualizations and explanations.
""")

# Sidebar for user input and model selection
st.sidebar.header('User Input Features')

def user_input_features():

    # File uploader for users to upload their own dataset
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read the uploaded CSV file into a pandas DataFrame
        df = pd.read_csv(uploaded_file)

        # Check if the required columns are present
        expected_columns = [
            'Elevation', 'Aspect', 'Slope', 
            'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways',
            'Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal Distance To Fire Points',
            'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4',
            'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4',
            'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8',
            'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12',
            'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16',
            'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',
            'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',
            'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',
            'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',
            'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',
            'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40'
        ]

        # Ensure all expected columns are in the uploaded DataFrame
        if all(col in df.columns for col in expected_columns):
            st.write("Uploaded CSV File:")
            st.write(df)
            return df
        else:
            st.warning("Uploaded file does not contain the required columns. Please check the format.")
            return None

    else:
        st.write("Awaiting CSV file upload. Alternatively, use the sliders and dropdowns to input data.")

        # Fallback to manual input if no file is uploaded
        Elevation = st.sidebar.slider('Elevation', 0, 4000, 2000)
        Aspect = st.sidebar.slider('Aspect', 0, 360, 180)
        Slope = st.sidebar.slider('Slope', 0, 60, 30)
        Horizontal_Distance_To_Hydrology = st.sidebar.slider('Horizontal Distance To Hydrology', 0, 5000, 2000)
        Vertical_Distance_To_Hydrology = st.sidebar.slider('Vertical Distance To Hydrology', -500, 1000, 0)
        Horizontal_Distance_To_Roadways= st.sidebar.slider('Horizontal_Distance_To_Roadways',0, 5000, 2000)

        # Hillshade features
        Hillshade_9am = st.sidebar.slider('Hillshade at 9am', 0, 255, 100)
        Hillshade_Noon = st.sidebar.slider('Hillshade at Noon', 0, 255, 100)
        Hillshade_3pm = st.sidebar.slider('Hillshade at 3pm', 0, 255, 100)
        
        Horizontal_Distance_To_Fire_Points = st.sidebar.slider('Horizontal Distance To Fire Points', 0, 5000, 2000)

        # Initialize wilderness area and soil type features
        wilderness_area = st.sidebar.selectbox('Wilderness_Area', ['Area 1', 'Area 2', 'Area 3', 'Area 4'])
        soil_type = st.sidebar.selectbox('Soil_Type', [f'Type {i}' for i in range(1, 41)])

        # Create a dictionary to hold input data
        data = {
            'Elevation': Elevation,
            'Aspect': Aspect,
            'Slope': Slope,
            'Horizontal_Distance_To_Hydrology': Horizontal_Distance_To_Hydrology,
            'Vertical_Distance_To_Hydrology': Vertical_Distance_To_Hydrology,
            'Horizontal_Distance_To_Roadways':Horizontal_Distance_To_Roadways,
            'Hillshade_9am': Hillshade_9am,
            'Hillshade_Noon': Hillshade_Noon,
            'Hillshade_3pm': Hillshade_3pm,
            'Horizontal_Distance_To_Fire_Points': Horizontal_Distance_To_Fire_Points,
        }

        # One-hot encode wilderness areas
        for i in range(1, 5):
            data[f'Wilderness_Area{i}'] = 1 if wilderness_area == f'Area {i}' else 0

        # One-hot encode soil types
        for i in range(1, 41):
            data[f'Soil_Type{i}'] = 1 if soil_type == f'Type {i}' else 0

        # Convert to DataFrame
        features = pd.DataFrame(data, index=[0])

        return features

# Get user input (either from uploaded file or manual input)
df_user = user_input_features()

# Display user inputs
st.write("User Input Features")
st.write(df_user)

# Preprocess user input
def preprocess_input(df_user):
    # Extract columns representing wilderness areas and soil types
    # wilderness_columns = [col for col in df_user.columns if 'Wilderness_Area' in col]
    # soil_columns = [col for col in df_user.columns if 'Soil_Type' in col]
    
    # Combine the wilderness and soil types for categorical features
    # cat_features = wilderness_columns + soil_columns
    
    # Select numeric features
    # numeric_features = df_user.drop(cat_features, axis=1)
 
    # Combine numeric features and encoded categorical features
    X = df_user
   
    # Generate interaction terms
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(X)
    X_poly_df = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(X.columns))
    
    return X_poly_df

X_user = preprocess_input(df_user)

# Model selection
selected_model = st.sidebar.selectbox('Select Model', list(models.keys()))

# Make prediction
prediction = models[selected_model].predict(X_user)
prediction_proba = models[selected_model].predict_proba(X_user)

# Display results
st.subheader('Prediction')
cover_types = ['Spruce/Fir', 'Lodgepole Pine', 'Ponderosa Pine', 'Cottonwood/Willow', 'Aspen', 'Douglas-fir', 'Krummholz']
st.write(f"The predicted forest cover type is: {cover_types[prediction[0] - 1]}")

st.subheader('Prediction Probability')
prob_df = pd.DataFrame(prediction_proba, columns=cover_types)
st.write(prob_df)

# Visualize prediction probabilities
st.subheader('Prediction Probability Visualization')
fig, ax = plt.subplots()
prob_df.T.plot(kind='bar', ax=ax)
plt.title('Prediction Probabilities for Each Cover Type')
plt.xlabel('Cover Type')
plt.ylabel('Probability')
st.pyplot(fig)

# Feature importance
st.subheader('Feature Importance')
if hasattr(models[selected_model], 'feature_importances_'):
    importances = models[selected_model].feature_importances_
    feature_importance = pd.DataFrame({
        'feature': X_user.columns,
        'importance': importances
    }).sort_values('importance', ascending=False)

    fig, ax = plt.subplots()
    sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
    plt.title('Top 10 Important Features')
    st.pyplot(fig)
else:
    st.write("Feature importance is not available for this model.")

# SHAP values for model interpretation
st.subheader('SHAP Value Interpretation')
explainer = shap.TreeExplainer(models[selected_model])
shap_values = explainer.shap_values(X_user)

st.write("SHAP summary plot")
shap.summary_plot(shap_values, X_user, plot_type="bar")
st.pyplot(plt.gcf())

# Dataset overview
st.subheader('Dataset Overview')
st.write(df.describe())

# Correlation heatmap
st.subheader('Feature Correlation Heatmap')

wilderness_columns = [col for col in df.columns if 'Wilderness_Area' in col]
soil_columns = [col for col in df.columns if 'Soil_Type' in col]
columns_to_drop = ['Cover_Type'] + wilderness_columns + soil_columns+['Id']
corr = df.drop(columns=columns_to_drop, axis=1).corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Distribution of cover types
st.subheader('Distribution of Cover Types')
fig, ax = plt.subplots()
df['Cover_Type'].value_counts().plot(kind='bar', ax=ax)
plt.title('Distribution of Cover Types')
plt.xlabel('Cover Type')
plt.ylabel('Count')
st.pyplot(fig)