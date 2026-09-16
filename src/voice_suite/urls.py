from django.contrib.auth import views as auth_views
from django.urls import include, path

urlpatterns = [
    path(
        "accounts/login/",
        auth_views.LoginView.as_view(template_name="dashboard/login.html"),
        name="login",
    ),
    path(
        "accounts/logout/",
        auth_views.LogoutView.as_view(next_page="login"),
        name="logout",
    ),
    path("", include("dashboard.urls")),
]
