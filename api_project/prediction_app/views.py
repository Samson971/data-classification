import json

import joblib
import numpy as np
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from rest_framework.parsers import JSONParser

from .models import Activity
from .serializers import ActivitySerializer
activity_labels = {
1  : 'WALKING',
2  : 'WALKING_UPSTAIRS',
3  : 'WALKING_DOWNSTAIRS',
4  : 'SITTING',
5  : 'STANDING',
6  : 'LAYING',
7  : 'STAND_TO_SIT',
8  : 'SIT_TO_STAND',
9  : 'SIT_TO_LIE',
10 : 'LIE_TO_SIT',
11 : 'STAND_TO_LIE',
12 : 'LIE_TO_STAND',
}

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
    #if request.method = = 'POST':
        model            = joblib.load('/home/samson971/Documents/Python_For_Data/data-project/Models/decision_tree_classifer_model.sav')
        body             = json.loads(request.body.decode("utf-8"))
        test             = body['data']
        #content         = body['content']
        #data            = JSONParser().parse(body['data'])
        #activity_data   = ActivitySerializer(data=body)
   #if activity_data.is_valid():
        activity_ids = model.predict(test)
        
        results = [{'id':int(act_id),'label': activity_labels[act_id]} for act_id in activity_ids]  
        return JsonResponse({'results':results})
