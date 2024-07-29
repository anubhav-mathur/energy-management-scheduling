from django.conf.urls import include,url
from . import views
from django.urls import path

app_name='major'

urlpatterns = [
    #url(r'^$', views.IndexView.as_view(), name="index"),

    url(r'^$',views.index,name='index'),
    url('upload-data',views.detail, name='detail'),
    url('uploadsuccess',views.uploadsuccess,name='uploadsuccess'),
    path('result', views.result, name="result"),
    url('monitor',views.monitor,name="monitor"),
    url('moutput1',views.moutput1,name='moutput1'),
    url('moutput2',views.moutput2,name='moutput2'),
    url('moutput3',views.moutput3,name='moutput3'),
    url('moutput4',views.moutput4,name='moutput4'),
    url('moutput5',views.moutput5,name='moutput5'),
    url('moutput6',views.moutput6,name='moutput6')

]
