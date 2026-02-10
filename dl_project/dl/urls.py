from django.urls import path
from . import views
app_name="dl"

urlpatterns=[
  path("",views.index , name="index"),
  path("add",views.add , name="add"),
  path('file/<uuid:request_uuid>/', views.file, name='file'),
  path('edit/<uuid:request_uuid>/', views.edit, name='edit'),
] 