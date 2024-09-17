import pandas as pd
import joblib

def load_test_data():
    return pd.read_csv('data/test.csv')

def preprocess_test_data(data):
    # Apply the same preprocessing steps as in preprocess.py
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
    data['Fare'].fillna(data['Fare'].median(), inplace=True)

    # Encode categorical columns
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    data = pd.get_dummies(data, columns=['Embarked'], drop_first=True)

    # Drop unnecessary columns
    data = data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

    return data

def make_predictions():
    # Load the test data
    test_data = load_test_data()

    # Preprocess the test data
    test_data_processed = preprocess_test_data(test_data)

    # Load the trained model
    model = joblib.load('models/logistic_model.pkl')

    # Make predictions
    predictions = model.predict(test_data_processed)

    # Prepare the submission dataframe
    submission = pd.DataFrame({
        'PassengerId': test_data['PassengerId'],
        'Survived': predictions
    })

    # Save the submission file
    submission.to_csv('submission/submission.csv', index=False)

if __name__ == '__main__':
    make_predictions()
