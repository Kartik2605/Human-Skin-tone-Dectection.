from django.conf.urls import url
from django.urls import path
from skintone_app import views


app_name = 'skintone_app'

'''
urlpatterns = [
    url(r'^register/$',views.register,name='register'),
    url(r'^user_login/$',views.user_login,name='user_login'),
]
'''

urlpatterns = [
    path('upload/', views.image_upload_view),
]