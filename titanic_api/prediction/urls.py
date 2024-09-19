from django.urls import path
from . import views  # Ensure you are importing views from the same directory

urlpatterns = [
    path('predict/', views.predict, name='predict'),  # This maps to /api/predict/
]
