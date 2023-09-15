from django.db import models
import os
from django.conf import settings
# Create your models here.

class FaceModel(models.Model):
    user_id = models.UUIDField(unique=True)
    first_name = models.CharField(max_length=50)
    last_name = models.CharField(max_length=50)
    email = models.EmailField(unique=True)
    h5_model = models.FileField(upload_to=os.path.join('./h5_folder'))
    face_labels = models.JSONField()
    headshot_directory = models.CharField(max_length=255, null=True, blank=True)

