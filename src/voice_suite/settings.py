"""Django settings for the voice_suite project skeleton (issue #19).

Mirrors clinical-llm-suite's / document-ai-suite's settings pattern —
SQLite by default with no external service required, DATABASE_URL
overrides for Postgres in deployment.
"""
import os
from pathlib import Path
from urllib.parse import urlparse

from django.core.exceptions import ImproperlyConfigured

BASE_DIR = Path(__file__).resolve().parents[2]

SECRET_KEY = "local-development-only"
DEBUG = True
ALLOWED_HOSTS = ["127.0.0.1", "localhost", "testserver"]


def _database_config_from_env():
    """Build the DATABASES["default"] entry from DATABASE_URL.

    Unset/empty DATABASE_URL is the supported default (local SQLite). A
    DATABASE_URL that *is* set must parse to a real Postgres connection, or
    startup fails loudly instead of silently falling back to SQLite.
    """
    raw_url = os.environ.get("DATABASE_URL", "").strip()
    if not raw_url:
        return {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": BASE_DIR / "db.sqlite3",
        }

    parsed = urlparse(raw_url)
    if parsed.scheme not in ("postgres", "postgresql"):
        raise ImproperlyConfigured(
            f"DATABASE_URL has unsupported scheme {parsed.scheme!r}; "
            "expected 'postgres://' or 'postgresql://', or unset it to use SQLite."
        )
    if not parsed.hostname or not parsed.path.lstrip("/"):
        raise ImproperlyConfigured(
            "DATABASE_URL is missing a host or database name "
            "(expected postgres://user:password@host:port/dbname)."
        )

    return {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": parsed.path.lstrip("/"),
        "USER": parsed.username or "",
        "PASSWORD": parsed.password or "",
        "HOST": parsed.hostname,
        "PORT": parsed.port or "",
    }


INSTALLED_APPS = [
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.staticfiles",
    "dashboard",
]
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
]
ROOT_URLCONF = "voice_suite.urls"
TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
            ]
        },
    }
]
WSGI_APPLICATION = "voice_suite.wsgi.application"

DATABASES = {"default": _database_config_from_env()}

LOGIN_URL = "login"
LOGIN_REDIRECT_URL = "dashboard"

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_TZ = True
STATIC_URL = "static/"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
