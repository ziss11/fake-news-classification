FROM tensorflow/serving:2.8.0

COPY ./serving_model_dir /models
ENV MODEL_NAME=fake-news-detection-model