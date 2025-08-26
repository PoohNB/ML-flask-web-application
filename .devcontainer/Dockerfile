# Use the official Python image as the base image
FROM python:3.8-slim AS base
# Set the working directory in the container
WORKDIR /APP
# Copy the requirements file into the container
COPY requirements.txt .
# Install required packages using pip
RUN pip install --no-cache-dir -r requirements.txt


FROM base AS dev 
# Install required packages using pip
RUN pip install pytest bandit safety


FROM python:3.8-slim AS prod

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /APP
COPY --from=base /usr/local /usr/local  
COPY . .
EXPOSE 90
CMD ["gunicorn","-c","gunicorn/gunicorn.conf.py","main:app"]
