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

Install the necessary dependencies using the following:

```bash
pip install torch pandas numpy scikit-learn ta-lib matplotlib

Model Overview

The model consists of a simple feed-forward neural network with the following architecture:

    Input Layer: 6 features (Open, Close, High, Low, MA10, RSI).
    Hidden Layer: 6 units, with a Tanh activation function.
    Output Layer: 2 units (representing the probability of the stock price going up or down), using a Softmax activation.

The model uses the CrossEntropyLoss criterion, which is suitable for classification problems with categorical outputs, and the Adam optimizer for training.
