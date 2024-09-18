INSTALLED_APPS = [
    'rest_framework',
    'prediction',
    'corsheaders',
]
MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',  # This should be at the top
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

# Allow requests from your React app
CORS_ALLOWED_ORIGINS = [
    'http://localhost:3000',  # Your React app's address
]

# For development, you can also use:
CORS_ALLOW_ALL_ORIGINS = True
