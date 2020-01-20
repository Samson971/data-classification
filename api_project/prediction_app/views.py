from django.http import HttpResponse, JsonResponse
from django.shortcuts import render

from rest_framework.parsers import JSONParser

from .models import Activity
from .serializers import ActivitySerializer


# Create your views here.
def prediction(request):
    return HttpResponse('My first prediction')

def save_an_activity(request):
    activity = Activity(
        
    )

def count(request):
    return HttpResponse( len(Activity.objects.all()))

def activity_detail(request,activity_id):
    activity   = Activity.objects.get(pk=activity_id)
    serializer = ActivitySerializer(activity)
    return JsonResponse(serializer.data)

def predict (request,model='decision_tree_classifier'):
    from sklearn.externals import joblib
    model           = joblib.load('/home/samson971/Documents/Python_For_Data/data-project/Models/decision_tree_classifiermodel.sav')
    data        = JSONParser().parse(request)
    serializer  = ActivitySerializer(data=data)
    #if serializer.is_valid():

