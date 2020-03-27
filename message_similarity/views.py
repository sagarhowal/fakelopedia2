from django.shortcuts import render
from django.shortcuts import render
from django.http import Http404
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from django.core import serializers
from django.conf import settings
import json

# Create your views here.
from message_similarity.similarity import cosine_sim


@api_view(["POST"])
def ideal_number(number):
    try:
        number = json.loads(number.body)
        big_number = str(number * 10)
        return JsonResponse("Ideal number should be: " + big_number, safe=False)
    except ValueError as e:
        return Response(e.args[0], status.HTTP_400_BAD_REQUEST)


@api_view(["POST"])
def cosine_similarity(messages_data):
    try:
        message = json.loads(messages_data.body)
        message_1 = message["message1"]
        message_2 = message["message2"]
        sim_score = cosine_sim(message_1, message_2)
        return JsonResponse(sim_score, safe=False)
    except ValueError as e:
        return Response(e.args[0], status.HTTP_400_BAD_REQUEST)


