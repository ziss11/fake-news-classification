FROM tensorflow/serving:latest

COPY ./serving_model /models
ENV MODEL_NAME=fake-news-detection-model