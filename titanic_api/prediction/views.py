# prediction/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import joblib
import pandas as pd

class PredictView(APIView):
    def post(self, request, *args, **kwargs):
        data = request.data
        # Load the trained model
        model = joblib.load('models/logistic_model.pkl')
        
        # Preprocess data
        df = pd.DataFrame(data)
        df['Age'].fillna(df['Age'].median(), inplace=True)
        df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
        df['Fare'].fillna(df['Fare'].median(), inplace=True)
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
        df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
        df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
        
        # Make predictions
        predictions = model.predict(df)
        
        # Prepare response
        return Response({'predictions': predictions.tolist()}, status=status.HTTP_200_OK)
