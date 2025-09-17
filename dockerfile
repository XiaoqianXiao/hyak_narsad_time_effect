# Use Ubuntu 20.04 as the base image
FROM ubuntu:20.04

# Set non-interactive frontend to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    wget \
    bzip2 \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    git \
    pkg-config \
    libhdf5-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install FSL 6.0.4
RUN wget https://fsl.fmrib.ox.ac.uk/fsldownloads/fsl-6.0.4-centos6_64.tar.gz \
    && tar -xzf fsl-6.0.4-centos6_64.tar.gz -C /usr/local \
    && rm fsl-6.0.4-centos6_64.tar.gz

# Set FSL environment variables
ENV FSLDIR=/usr/local/fsl
ENV PATH=${FSLDIR}/bin:${PATH}
ENV FSLOUTPUTTYPE=NIFTI_GZ

# Install Python dependencies
RUN pip3 install --upgrade pip \
    && pip3 install \
        bids \
        nipype \
        pandas \
        templateflow \
        numpy

# Set working directory
WORKDIR /app

# Copy the script and workflows file into the container
COPY run_1st_level.py /app/run_1st_level.py
COPY workflows.py /app/workflows.py  # Remove this line if workflows.py isn’t separate

# Ensure the script is executable
RUN chmod +x /app/run_1st_level.py

# Set the entrypoint to run the script
ENTRYPOINT ["python3", "/app/run_1st_level.py"]

# Default command (can be overridden)
CMD []