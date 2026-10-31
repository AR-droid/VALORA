"""Gunicorn entry point: ``gunicorn wsgi:app``."""

from valora.web.app import create_app

app = create_app()
