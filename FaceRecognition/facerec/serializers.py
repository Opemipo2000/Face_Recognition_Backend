from rest_framework import serializers
from .models import FaceModel

class FaceModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = FaceModel
        fields = '__all__'
