FROM tensorflow/serving:latest

COPY ./serving_model_dir /models
ENV MODEL_NAME=fake-news-detection-model