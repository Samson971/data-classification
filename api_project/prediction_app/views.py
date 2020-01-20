import json

import joblib
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from rest_framework.parsers import JSONParser

from .models import Activity
from .serializers import ActivitySerializer


# Create your views here.
def prediction(request):
    return HttpResponse('My first prediction')

def count(request):
    return HttpResponse( len(Activity.objects.all()))

def activity_detail(request,activity_id):
    activity   = Activity.objects.get(pk=activity_id)
    serializer = ActivitySerializer(activity)
    return JsonResponse(serializer.data)

@csrf_exempt
def predict (request,model='decision_tree_classifier'):
    if request.method == 'POST':
        model           = joblib.load('/home/samson971/Documents/Python_For_Data/data-project/Models/decision_tree_classifer_model.sav')
        body = json.loads(request.body.decode('utf-8'))
        #content = body['content']
        #data        = JSONParser().parse(body['data'])
        activity_data = ActivitySerializer(data=body)
        if activity_data.is_valid():
            label = model.predict(activity_data)
            results = {}
            results['label'] = label
            return HttpResponse(body)
