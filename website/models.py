from django.db import models
from django.db.models import IntegerField, Model, DateTimeField, CharField,IntegerField
from django_mysql.models import ListTextField


# Create your models here.
class Prediction(models.Model):
    solar=models.FileField()
    wind=models.FileField()
    house=models.FileField()
    resi=models.IntegerField(max_length=2, null=True, blank=True, default=None)
    coal=ListTextField(base_field=IntegerField(),size=100,)
    store=ListTextField(base_field=IntegerField(),size=100,)
    window=models.CharField(max_length=3)
    non_renewable=models.CharField(max_length=3)
    ddate=models.DateTimeField( null=True,blank=True)

    def get_month(self):
        return self.ddate.strftime('%m')
