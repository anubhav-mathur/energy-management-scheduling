# Generated by Django 2.1 on 2020-04-28 15:18

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('major', '0003_prediction_resi'),
    ]

    operations = [
        migrations.AlterField(
            model_name='prediction',
            name='resi',
            field=models.CharField(max_length=3),
        ),
    ]