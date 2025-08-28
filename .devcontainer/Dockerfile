# Use the official Python image as the base image
FROM python:3.8-slim AS base


WORKDIR /APP

ENV PYTHONPATH="/APP"

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


FROM base AS dev 
# Install required packages using pip
COPY . .
RUN pip install pytest bandit safety


FROM python:3.8-slim AS prod

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /APP
COPY --from=base /usr/local /usr/local 
COPY . .
CMD ["gunicorn","-c","configs/gunicorn.conf.py","wsgi:app"]
