import pandas as pd
import joblib
import os

def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'model_files', 'logistic_model.pkl')
    return joblib.load(model_path)

def preprocess_data(data):
    # Same preprocessing steps
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
    data['Fare'].fillna(data['Fare'].median(), inplace=True)
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    data = pd.get_dummies(data, columns=['Embarked'], drop_first=True)
    data = data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], errors='ignore')
    return data

def make_prediction(input_data):
    model = load_model()
    processed_data = preprocess_data(pd.DataFrame([input_data]))
    prediction = model.predict(processed_data)
    return prediction[0]  # return 0 or 1
