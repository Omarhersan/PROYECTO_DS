# Use the Python 3.10 slim-buster image as the base image
FROM python:3.10-slim-buster
LABEL authors="kaleb.avila, omar.hernandez, axel.jaramillo"

# Set the working directory inside the container
WORKDIR /app

# Install necessary packages and configure timezone
RUN apt-get update && apt-get install -y procps && pip install -U pip \
    && rm /etc/localtime \
    && ln -s /usr/share/zoneinfo/America/Mexico_City /etc/localtime  \
    && apt-get install libgomp1


RUN apt-get update --yes && \
    apt-get upgrade --yes && \
    apt-get install --yes --no-install-recommends \
    # - apt-get upgrade is run to patch known vulnerabilities in apt-get packages as
    #   the python base image may be rebuilt too seldom sometimes (less than once a month)
    # required for psutil python package to install
    python3-dev \
    gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


# Copy the requirements file into the container and install dependencies
COPY ./requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

COPY ./main.py /main.py
COPY ./best.pt /best.pt

# Expose port 8000
EXPOSE 8000

# Set the command to run FastAPI using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]