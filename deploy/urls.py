from django.urls import path, include
from deploy import views
urlpatterns = [
    path('home', views.home, name='home'),
    path('result', views.result, name='result')
]
