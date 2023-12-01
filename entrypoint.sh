#!/bin/bash

# Function to handle the SIGINT signal (Ctrl-C)
cleanup() {
    echo "SIGINT caught, stopping processes..."

    # Kill the TensorBoard process
    kill $TENSORBOARD_PID

    # Kill the training process
    kill $TRAIN_PID

    # Optionally, wait for the processes to stop
    wait $TENSORBOARD_PID
    wait $TRAIN_PID

    echo "Processes stopped. Exiting."
    exit 0
}

# Set trap to catch SIGINT and call the cleanup function
trap cleanup SIGINT

# Check the first argument to determine the mode
if [ "$1" = "train" ]; then
    # Remove the first argument ('train')
    shift 

    # Run training command with all remaining arguments in the background
    python tools/train.py "$@" &
    TRAIN_PID=$!

    # Start TensorBoard in the background
    tensorboard --logdir=runs --host=0.0.0.0 &
    TENSORBOARD_PID=$!

    # Wait for the training process to complete
    wait $TRAIN_PID

elif [ "$1" = "infer" ]; then
    # Run inference command
    # Assuming you might also want to pass arguments to infer.py in the future
    shift
    python tools/infer.py "$@"
else
    echo "Invalid argument. Please use 'train' or 'infer'."
fi
