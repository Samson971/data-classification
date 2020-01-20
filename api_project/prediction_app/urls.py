from django.contrib import admin
from django.urls import path

from prediction_app import views

urlpatterns = [
    path('test/', views.prediction,name='prediction'),
    path('count/', views.count,name='count'),
    path('activity/<int:activity_id>',views.activity_detail,name='activity_detail')
]
