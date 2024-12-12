# Stock Price Prediction with Neural Network

This project demonstrates how to predict stock price movements (up/down) using a simple neural network. The model uses historical stock data with technical indicators, such as moving averages and RSI (Relative Strength Index), to predict whether the stock price will go up or down on the following day.

## Project Structure

- **NeuralNet Class**: Defines a feed-forward neural network with one hidden layer.
- **Data Preprocessing**: The stock price data is cleaned, transformed, and normalized before being fed into the model.
- **Model Training**: The neural network is trained using CrossEntropyLoss and an Adam optimizer.
- **Model Evaluation**: After training, the model's performance is evaluated on a test set using accuracy, precision, recall, and F1 score metrics.

## Dependencies

This project uses the following Python libraries:

- `torch` - For building and training the neural network.
- `pandas` - For data manipulation and preprocessing.
- `numpy` - For numerical operations.
- `scikit-learn` - For data scaling and splitting the dataset.
- `ta-lib` (optional) - For technical analysis indicators (e.g., RSI).
- `matplotlib` (optional) - For plotting results.

Model Overview

The model consists of a simple feed-forward neural network with the following architecture:

    Input Layer: 6 features (Open, Close, High, Low, MA10, RSI).
    Hidden Layer: 6 units, with a Tanh activation function.
    Output Layer: 2 units (representing the probability of the stock price going up or down), using a Softmax activation.

The model uses the CrossEntropyLoss criterion, which is suitable for classification problems with categorical outputs, and the Adam optimizer for training.
Data Preprocessing

The dataset used for training consists of historical stock price data with the following steps:

Data Cleaning: Remove dollar signs and commas from the numerical columns (Open, Close, High, Low).
Feature Engineering:
    Moving Average (MA10): A 10-day rolling average of the closing price.
    RSI (Relative Strength Index): A momentum indicator used to measure the speed and change of price movements.
Scaling: The data is scaled using StandardScaler to normalize the features.
Label Creation: A binary classification label is created based on whether today's closing price is higher than tomorrow's

Evaluation Results:

Accuracy: 71.46%
Precision: 0.7228
Recall: 0.7878
F1 Score: 0.7539


How to Run the Code

Clone this repository:

    git clone https://github.com/yourusername/stock-price-prediction.git
    cd stock-price-prediction

Prepare your dataset and place it in the appropriate folder (if applicable).

Run the notebook or Python script:

    python train_model.py
