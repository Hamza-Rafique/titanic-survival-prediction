import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

def load_preprocessed_data():
    return pd.read_csv('data/preprocessed_train.csv')

def train_model():
    data = load_preprocessed_data()
    X = data.drop(columns=['Survived'])
    y = data['Survived']
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Validate the model
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {accuracy}")

    # Save the model
    joblib.dump(model, 'models/logistic_model.pkl')

if __name__ == '__main__':
    train_model()
