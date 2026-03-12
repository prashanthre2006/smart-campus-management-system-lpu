import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "dev-secret-key-change-me")
DEBUG = os.getenv("DJANGO_DEBUG", "true").lower() in {"1", "true", "yes", "on"}
ALLOWED_HOSTS = [h.strip() for h in os.getenv("DJANGO_ALLOWED_HOSTS", "*").split(",") if h.strip()]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.messages",
    "django.contrib.sessions",
    "django.contrib.staticfiles",
    "webapp",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
]

ROOT_URLCONF = "face_site.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "webapp" / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ]
        },
    }
]

WSGI_APPLICATION = "face_site.wsgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "data" / "attendance.db",
    }
}

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = False
USE_TZ = False

STATIC_URL = "/static/"
STATICFILES_DIRS = [BASE_DIR / "webapp" / "static"]

# Auth
LOGIN_URL = "/login"
LOGIN_REDIRECT_URL = "/dashboard"

# Django 5 default
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Use signed cookies to avoid DB session tables
SESSION_ENGINE = "django.contrib.sessions.backends.signed_cookies"
SESSION_COOKIE_AGE = int(os.getenv("SESSION_COOKIE_AGE", "3600"))  # 60 minutes by default
SESSION_EXPIRE_AT_BROWSER_CLOSE = os.getenv("SESSION_EXPIRE_AT_BROWSER_CLOSE", "false").lower() in {"1", "true", "yes"}
SESSION_SAVE_EVERY_REQUEST = True

CSRF_COOKIE_SECURE = os.getenv("CSRF_COOKIE_SECURE", "false").lower() in {"1", "true", "yes"}
SESSION_COOKIE_SECURE = os.getenv("SESSION_COOKIE_SECURE", "false").lower() in {"1", "true", "yes"}
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
