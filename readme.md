## Purpose

This project is a simple FastAPI project that demonstrates how to use FastAPI to create a simple REST API for waste detection.

## Installation

Make sure your system has the following dependencies installed:
- python3 >= 3.10
- pip3 >= 22.2.2

I highly recommend to use pycharm as the IDE for this project since we need to set up many things.
I also recommend to use python virtual environment for this project to avoid conflicts with other project dependencies.
To install the package, run the following command:

```pip install -r requirements.txt```

## Data Training and Setup (optional)

The project already has a pre-trained model that is ready to use.

In case you wish to train your own model, fill the trainable_data folder with your dataset, then you can use the following command:

```python app/train.py```

For more information about how to set up the training data dependencies, hit link below:
https://github.com/entbappy/Setup-NVIDIA-GPU-for-Deep-Learning?tab=readme-ov-file

## Usage

To run the project, execute command below:

```fastapi dev app/main.py```

The server will be running on `http://127.0.0.1:8000

## Endpoints

List of the available endpoints are accessible on `http://127.0.0.1:8000/docs`