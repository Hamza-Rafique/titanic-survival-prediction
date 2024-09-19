from django.urls import path
from . import views  # Assuming your views are in the same app

urlpatterns = [
    path('api/predict/', views.predict, name='predict'),  # This maps to the 'predict' view
]
