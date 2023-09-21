# Use the official Python image as the base image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /APP

# Copy the requirements file into the container
COPY requirements.txt .

# Install required packages using pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Command to run the application
CMD ["python", "app.py"]
