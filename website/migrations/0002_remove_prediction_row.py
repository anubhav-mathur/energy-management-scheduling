# Generated by Django 2.1 on 2020-04-28 14:21

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('major', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='prediction',
            name='row',
        ),
    ]