from django.urls import path
from .views import process_video

urlpatterns = [
  #  path('', camera, name='camera'),
  #  path('', Home, name='Home'),
    path('', process_video , name = 'process_video')
]

# from django.urls import path
# from . import views


# urlpatterns = [
#     path('', views.index, name='index'),
#     path('video_gen', views.video_gen, name='video_gen'),
#     #3path('webcam_feed', views.webcam_feed, name='webcam_feed'),
 
#     ]