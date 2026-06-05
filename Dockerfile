# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app
ENV PYTHONPATH=/app


# Install system dependencies required for audio/image processing libraries
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
# using --no-cache-dir to keep the image size down
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Define environment variable for unbuffered output (useful for logging)
ENV PYTHONUNBUFFERED=1

# Expose ports for MLFlow (5000) and Streamlit (8501)
EXPOSE 5000 8501

# Run the application
CMD ["python", "main.py"]
