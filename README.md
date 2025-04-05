# ASSIGNMENT-3--Q4
Sentiment Classification Using RNN
This project involves training a sentiment analysis model using the IMDB movie reviews dataset. The task is to classify the sentiment of movie reviews as either positive or negative based on the provided text using a Long Short-Term Memory (LSTM) model.
Name: Sony Pailla

Course: CS5720 Neural Network and Deep Learning

University: University of Central Missouri

GitHub: sonypailla

Date: April 2025
Steps
1. Load the IMDB Sentiment Dataset
The dataset used in this task is the IMDB dataset, which contains movie reviews labeled as positive or negative. The dataset is loaded from tensorflow.keras.datasets.imdb.
2. Preprocess the Text Data
The dataset consists of integer-encoded sequences, where each integer represents a word in the review. The text data is preprocessed by:

Padding Sequences to make all input sequences have the same length.

Tokenization is performed by using Keras's built-in tokenizer to convert words into a fixed-length sequence of integers.
3. Train an LSTM-based Model
The model is a sequential neural network consisting of an embedding layer, an LSTM layer, and a dense layer. The LSTM layer captures the sequential nature of text data, which helps in sentiment classification
4. Generate Confusion Matrix and Classification Report
After training the model, we evaluate its performance using a confusion matrix and a classification report. These metrics provide insights into the model's accuracy, precision, recall, and F1-score.
5. Precision-Recall Tradeoff in Sentiment Classification
Precision and recall are critical metrics in sentiment classification. Precision tells us how many of the positive predictions were actually positive, while recall tells us how many of the true positive labels we were able to correctly identify.

Precision-Recall Tradeoff: In sentiment analysis, a tradeoff exists between precision and recall. Optimizing for precision can reduce false positives but may miss some true positives, whereas optimizing for recall might identify more true positives but at the cost of more false positives. This tradeoff is important in real-world applications, where you might prefer one over the other depending on the cost of misclassification.
Requirements
TensorFlow

Keras

scikit-learn

Install the necessary dependencies via pip:
pip install tensorflow scikit-learn
Conclusion
This project demonstrates the process of building and evaluating an LSTM-based sentiment analysis model using the IMDB dataset. By generating a confusion matrix and classification report, we can assess the model's performance in terms of accuracy, precision, recall, and F1-score. Understanding the precision-recall tradeoff is crucial when fine-tuning the model for specific use cases in sentiment analysis.
