import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data():
    train_data = pd.read_csv('./data/Titanic-Dataset.csv')
    return train_data

def preprocess_data(data):
    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    data = pd.get_dummies(data, columns=['Embarked'], drop_first=True)
    data = data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
    return data

def scale_data(data):
    scaler = StandardScaler()
    data[['Age', 'Fare']] = scaler.fit_transform(data[['Age', 'Fare']])
    return data

if __name__ == '__main__':
    # Load and preprocess the data
    train_data = load_data()
    train_data = preprocess_data(train_data)
    train_data = scale_data(train_data)
    
    # Save preprocessed data if needed
    train_data.to_csv('data/preprocessed_train.csv', index=False)
