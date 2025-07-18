from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from detector import views

urlpatterns = [
    path("admin/", admin.site.urls),
     path('', views.detect, name='detect'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
