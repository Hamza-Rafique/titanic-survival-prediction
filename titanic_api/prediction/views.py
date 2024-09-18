from rest_framework.decorators import api_view
from rest_framework.response import Response
from .serializers import PassengerDataSerializer
from .predict import make_prediction

@api_view(['POST'])
def predict_survival(request):
    serializer = PassengerDataSerializer(data=request.data)
    if serializer.is_valid():
        prediction = make_prediction(serializer.validated_data)
        return Response({'survived': prediction})
    return Response(serializer.errors, status=400)
