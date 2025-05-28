FROM python:3.10

# Create a non-root user for security
RUN useradd -m -u 1000 user

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file and install dependencies as root
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the application code (including app.py and resume.pdf)
COPY . .

# Switch to the non-root user
USER user

# Expose the port that the application will run on
EXPOSE 8003

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8003", "--reload"]