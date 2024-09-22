**Forest Cover Type Classifier**

This project is a web application that uses machine learning to classify forest cover types based on cartographic variables.

**Features**

Predict forest cover type based on user input
Visualize the distribution of forest cover types
Display feature importance
Show pairplot of top features

**Installation**

1. Clone this repository:
Copy
```python
git clone https://github.com/yourusername/forest-classifier.git
cd forest-classifier
```
2. Install the required packages:
Copy
```python
pip install -r txt/requirements.txt
```
3.Run the data preparation script:
Copy
```python
python py/data_preparation.py
```
4.Train the model:
Copy
```python
python py/model_training.py
```

**Usage**

To run the Streamlit app locally:
Copy
```python
streamlit run streamlit_app.py
```
To run the Docker container:
Copy
```python
docker build -t forest-type-classifier-new .
docker run -p 8501:8501 forest-type-classifier-new

```
Then open your web browser and go to http://localhost:8501

**Project Structure**

`data/train.csv`: The full Covertype dataset

`py/`: Contains the Python scripts for data preparation, model training, and the Streamlit application.

    **data_preparation.py**: Script to download and preprocess the dataset.

    **model_training.py**: Script to train and save the machine learning models.

    **streamlit_app.py**: Main Streamlit application for the web interface.

`joblib/`: Stores the trained and compressed machine learning models.

    Each `*_model_compressed.joblib` file is a serialized, compressed version of a trained model.
    These files are loaded by the Streamlit app to make predictions.

    `poly_compressed.joblib`: A serialized PolynomialFeatures object used for feature engineering.

`Dockerfile`: Instructions for containerizing the app

`txt/`: Contains text files related to project dependencies and versioning.

    `requirements.txt`: List of Python dependencies required for the project.

    `sklearn_version.txt`: Specifies the version of scikit-learn used in the project.

**License**

MIT