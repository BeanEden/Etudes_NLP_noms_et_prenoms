# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    libomp-dev \
    git \
    pandoc \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir nbconvert nbformat
RUN pip install --no-cache-dir -r requirements.txt || \
    (sed -i 's/==/>=/g' requirements.txt && pip install --no-cache-dir -r requirements.txt)

# Download SpaCy model (just in case it wasn't installed via requirements)
RUN python -m spacy download fr_core_news_sm

# Copy the rest of the application code into the container
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=flask/app.py

# Expose the port the app runs on
EXPOSE 5000

# Default command: run the Flask app
CMD ["python", "flask/app.py"]
