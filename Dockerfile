FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /usr/src/app

# Allow the container to use the GPU
ENV NVIDIA_VISIBLE_DEVICES all

# Copy everything from the current directory to /usr/src/app in the container
COPY . .

# Install requirements
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Install the package
RUN pip install --editable .

# Allow the app to use tensorboard from runs directory
RUN chmod -R 777 runs

# Expose the TensorBoard port
EXPOSE 6006

# Make the entrypoint script executable
RUN chmod +x /usr/src/app/entrypoint.sh

# Set the entrypoint script to be executed
ENTRYPOINT ["/usr/src/app/entrypoint.sh"]
