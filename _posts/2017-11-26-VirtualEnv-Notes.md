---
layout: post
title: Python VirtualEnv Notes
---

Virtualenv is tool for creating isolated python environments

## Virtualenv for a Django project ##

*Pre-req 1:* `sudo pip install virtualenv`

1. Create a new virtual env in the same directory where your Django project will live
```
~$ virtualenv ~/<env-name>
```
2. Activate the virtualenv
```
~$ source ~/<env-name>/bin/activate
```
3. Pip install libraries for your project (including django)
```
(<env-name>)~$ pip install django==1.11.6
```
4. List packages installed:
```
(<env-name>)~$ pip freeze
Django==1.11.6
```
5. Exit virtualenv
```
(eb-virt) ~/$ deactivate
```