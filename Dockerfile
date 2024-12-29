# Use the official Python image as the base
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements file to leverage Docker's caching mechanism
COPY requirements.txt ./

# Install dependencies in the correct order
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir transformers sentence-transformers \
    && pip install --no-cache-dir Django==5.0.1 djangorestframework==3.14.0 django-cors-headers==4.3.1

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on (update if needed)
EXPOSE 8000

# Define the default command to run the application
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
