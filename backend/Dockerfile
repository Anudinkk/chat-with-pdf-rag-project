# Stage 1: Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir reduces image size, --upgrade ensures latest pip
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code from your host to your container at /app
COPY . .

# Make port 8000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV NAME World

# Run main.py when the container launches
# The application will be started using uvicorn within your main.py
CMD ["python", "main.py"]