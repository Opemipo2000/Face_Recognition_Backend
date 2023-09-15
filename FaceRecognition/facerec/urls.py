from django.urls import path
from . import views

urlpatterns = [
    path('register/', views.RegisterFace.as_view(), name='register-face'),
    # http://127.0.0.1:8000/api/register
    path('identify/', views.IdentifyFace.as_view(), name='identify-face'),
    # http://127.0.0.1:8000/api/identify
    path('score/', views.calculateAttentiveness.as_view(), name='engagement-score'),
    # http://127.0.0.1:8000/api/score
]
