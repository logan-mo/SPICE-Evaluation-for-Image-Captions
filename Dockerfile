# Use an official lightweight Python image
FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /app

# Install dependencies
RUN apt update && apt install -y \
    default-jdk \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Download and install Stanford CoreNLP
RUN wget http://nlp.stanford.edu/software/stanford-corenlp-4.5.0.zip && \
    unzip stanford-corenlp-4.5.0.zip && \
    rm stanford-corenlp-4.5.0.zip

# Set environment variable for CoreNLP
ENV CORENLP_HOME=/app/stanford-corenlp-4.5.0

# Install required Python libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Start Stanford CoreNLP server on container start
CMD ["java", "-mx4g", "-cp", "/app/stanford-corenlp-4.5.0/*", "edu.stanford.nlp.pipeline.StanfordCoreNLPServer", "-port", "9000", "-timeout", "15000"]
