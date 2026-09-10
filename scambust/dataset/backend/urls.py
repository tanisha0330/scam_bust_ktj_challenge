from django.urls import path

from . import views

urlpatterns = [
    path("", views.home),
    path("predict", views.predict_scam),
    path("check_number", views.check_number),
    path("analyze_call", views.analyze_call),
]
