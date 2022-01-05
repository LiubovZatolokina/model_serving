# Model Serving
Serve Text Classification LSTM model with an attention mechanism with TorchServe tool.

- Build and run

>docker-compose up --build

- Predict
>curl http://0.0.0.0:8080/predictions/lstm_attention -T sample.txt
