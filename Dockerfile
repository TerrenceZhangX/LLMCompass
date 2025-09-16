# Start with a base image that includes Miniconda to manage our environment
FROM continuumio/miniconda3

# Set the working directory in the container to /app
WORKDIR /app

# Create the conda environment
COPY . /app
RUN conda env create -f /app/environment.yml

# Do not rely on `source activate` in non-interactive shells; set PATH to the env's bin
ENV PATH /opt/conda/envs/llmcompass_ae/bin:$PATH

# Install lightweight Python deps for the API server inside the conda env
RUN /opt/conda/envs/llmcompass_ae/bin/pip install \
    fastapi \
    "uvicorn[standard]" \
    aiosqlite \
    requests

# Expose the port your app runs on and run uvicorn as entrypoint
EXPOSE 8000
# NOTE: CMD removed to allow interactive testing. Re-add the uvicorn CMD after verification.


