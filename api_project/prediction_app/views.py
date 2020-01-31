import json
import os

import joblib
import numpy as np
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from rest_framework.parsers import JSONParser

from .models import Activity
from .serializers import ActivitySerializer
from api_project.settings import BASE_DIR

activity_labels = {
    1: 'WALKING',
    2: 'WALKING_UPSTAIRS',
    3: 'WALKING_DOWNSTAIRS',
    4: 'SITTING',
    5: 'STANDING',
    6: 'LAYING',
    7: 'STAND_TO_SIT',
    8: 'SIT_TO_STAND',
    9: 'SIT_TO_LIE',
    10: 'LIE_TO_SIT',
    11: 'STAND_TO_LIE',
    12: 'LIE_TO_STAND',
}

# Create your views here.


def prediction(request):
    return HttpResponse('My first prediction')


def count(request):
    return HttpResponse(len(Activity.objects.all()))


def activity_detail(request, activity_id):
    activity = Activity.objects.get(pk=activity_id)
    serializer = ActivitySerializer(activity)
    return JsonResponse(serializer.data)


@csrf_exempt
def predict(request):
    body = json.loads(request.body.decode("utf-8"))
    model = body['model']
    model = joblib.load(
        os.path.join(BASE_DIR,'models/'+model+'_model.sav'))
    data = body['data']
    activity_ids = model.predict(data)

    results = [{'id': int(act_id), 'label': activity_labels[act_id]}
               for act_id in activity_ids]
    return JsonResponse({'results': results})
