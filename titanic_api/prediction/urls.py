from django.urls import path
from .views import predict_survival

urlpatterns = [
    path('predict/', predict_survival, name='predict-survival'),
]
