from django.urls import path
from .views import health_check, predict


urlpatterns = [
    path("health/", health_check),
    path("predict/", predict),
]