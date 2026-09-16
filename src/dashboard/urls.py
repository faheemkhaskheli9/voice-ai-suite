from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="dashboard"),
    path("features/<slug:slug>/", views.feature_stub, name="feature"),
]
