from rest_framework import serializers

class PassengerDataSerializer(serializers.Serializer):
    Pclass = serializers.IntegerField()
    Sex = serializers.CharField()
    Age = serializers.FloatField()
    SibSp = serializers.IntegerField()
    Parch = serializers.IntegerField()
    Fare = serializers.FloatField()
    Embarked = serializers.CharField()
