from django.db import models

# Create your models here.

class Image(models.Model):
    title = models.CharField(max_length=200,blank=True)
    image = models.ImageField(upload_to='user_images')

    def __str__(self):
        return self.title 
        #+ "" + str(self.image)