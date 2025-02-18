# Use an official Python runtime as a parent image.
FROM python:3.9-slim

# Set environment variables to ensure output is logged.
ENV PYTHONUNBUFFERED True

# Set the working directory.
WORKDIR /app

# Copy requirements and install them.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the application code.
COPY . .

# Expose the port Cloud Run expects (default is 8080).
EXPOSE 8080

# Define environment variable for Flask.
ENV FLASK_APP app.py

# Run the Flask app. Cloud Run sets the PORT environment variable automatically.
CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]
