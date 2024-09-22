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
pip install -r requirements.txt
```
3.Run the data preparation script:
Copy
```python
python data_preparation.py
```
4.Train the model:
Copy
```python
python model_training.py
```

**Usage**

To run the Streamlit app locally:
Copy
```python
streamlit run app.py
```
To run the Docker container:
Copy
```python
docker build -t forest-type-classifier .
docker run -p 8501:8501 forest-classifier

```
Then open your web browser and go to http://localhost:8501

**Project Structure**

data/train.csv: The full Covertype dataset
data_preparation.py: Script to download and preprocess the dataset
model_training.py: Script to train and save the machine learning model
streamlit_app.py: Main Streamlit application
Dockerfile: Instructions for containerizing the app
requirements.txt: List of Python dependencies

**License**

MIT