from django.http import JsonResponse

def predict(request):
    if request.method == 'POST':
        # Your prediction logic goes here
        return JsonResponse({'message': 'Prediction result'})
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=400)
