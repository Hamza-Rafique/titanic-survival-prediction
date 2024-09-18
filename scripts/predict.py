import pandas as pd
import joblib

def load_test_data():
    """Load the test dataset."""
    return pd.read_csv('data/test.csv')

def preprocess_test_data(data, train_columns):
    """Preprocess the test data and ensure it matches the features used during training."""
    
    # Handle missing values
    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())

    # Encode categorical variables
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

    # One-hot encode 'Embarked'
    data = pd.get_dummies(data, columns=['Embarked'], drop_first=True)

    # Drop unnecessary columns
    data = data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

    # Ensure the test data has the same columns as the training data
    for col in train_columns:
        if col not in data.columns:
            data[col] = 0  # Fill missing columns with 0
    data = data[train_columns]  # Ensure correct order of columns

    return data

def make_predictions():
    """Make predictions on the test dataset and save the results."""
    
    # Load the test data
    test_data = load_test_data()

    # Load the trained model and extract the feature names from the model
    model = joblib.load('models/logistic_model.pkl')
    train_columns = model.feature_names_in_  # Columns used during training

    # Preprocess the test data to match the training columns
    test_data_processed = preprocess_test_data(test_data, train_columns)

    # Make predictions using the preprocessed test data
    predictions = model.predict(test_data_processed)

    # Prepare the submission DataFrame with PassengerId and predicted 'Survived' values
    submission = pd.DataFrame({
        'PassengerId': test_data['PassengerId'],  # Retain PassengerId for submission
        'Survived': predictions  # Predicted survival status
    })

    # Save the submission as a CSV file
    submission.to_csv('submission/submission.csv', index=False)

    print("Predictions saved successfully to 'submission/submission.csv'.")

if __name__ == '__main__':
    make_predictions()
