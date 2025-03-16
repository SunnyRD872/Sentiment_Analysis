
# Sentiment Analysis on Amazon Reviews



## Overview

This project focuses on performing sentiment analysis on an Amazon review dataset. The goal is to classify reviews as either positive (1) or negative (0) using machine learning techniques. The project involves exploratory data analysis (EDA), text preprocessing, feature extraction, and model training and evaluation.


## Dataset

- Source: Amazon Reviews(https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews?select=train.csv)

- Size: Taken 100,000 reviews

- Labels:

-0: Negative sentiment

-1: Positive sentiment
## Steps Performed

1. Exploratory Data Analysis (EDA)
- Null Value Check: Identified and handled any null values in the dataset.

- Duplicate Value Check: Removed duplicate reviews to ensure data quality.

- Visualization:

-Bar graph to visualize the distribution of sentiments (1 and 0).

-Histogram to analyze the distribution of characters in the text.

-Word clouds to visualize the most frequent words in positive and negative reviews separately.

2. Text Preprocessing
- Converted text to lowercase.

- Removed punctuation, HTML tags, and extra spaces.

- Eliminated stop words to reduce noise in the text data.

3. Feature Extraction
- Used TF-IDF (Term Frequency-Inverse Document Frequency) for vectorization to convert text data into numerical features.

4. Train-Test Split
Split the dataset into training and testing sets:

- Training set: 80,000 reviews

- Testing set: 20,000 reviews

5. Model Training and Evaluation
- Model Used: Logistic Regression

- Results:

-Accuracy: 83.08%

-Classification Report:

          precision    recall  f1-score   support

       0       0.83      0.82      0.82      9688
       1       0.83      0.85      0.84     10312

accuracy                           0.83     20000

macro avg       0.83      0.83      0.83     20000

weighted avg       0.83      0.83      0.83     20000


## Requirements

- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, nltk, wordcloud
## Future Work

- Experiment with other machine learning models like Random Forest, SVM, or deep learning models (e.g., LSTM, BERT).

## Acknowledgments

This project relies on several open-source libraries and frameworks. Special thanks to the following:

- Kaggle: For providing the Amazon Reviews dataset used in this project.

- Pandas: For data manipulation and analysis, enabling efficient handling of the dataset.

- NumPy: For numerical computations and array operations.

- Matplotlib and Seaborn: For creating visualizations such as bar graphs, histograms, and heatmaps.

- Scikit-learn: For providing tools for feature extraction (TF-IDF), train-test splitting, and implementing the Logistic Regression model.

- NLTK: For stopwords removal and text preprocessing functionalities.

- WordCloud: For generating word clouds to visualize the most frequent words in positive and negative reviews.

- Jupyter Notebook: For providing an interactive environment to develop and document the project.
