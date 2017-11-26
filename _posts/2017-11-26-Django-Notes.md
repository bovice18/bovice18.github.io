---
layout: post
title: Django Notes
---

Django is a Python web framework that makes MVC, database driven websites easy to create and scale.

## Create Django Project ##

1. [Create a virtual env](/VirtualEnv-Notes) for your new Django project
2. Create a new Django project (insert project name instead of *ebdjango*):
```
(<virtual-env>)~$ django-admin startproject ebdjango
```
This will create the following project structure:
```
~/ebdjango
  |-- ebdjango
  |   |-- __init__.py
  |   |-- settings.py
  |   |-- urls.py
  |   `-- wsgi.py
  `-- manage.py
```
3. Run your site locally to test the initial setup:
```
(<virtual-env>) ~/ebdjango$ python manage.py runserver
```
Open http://127.0.0.1:8000/
