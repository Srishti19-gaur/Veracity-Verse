# Veracity-Verse-FakeNewsDetection
 VeracityVerse is a fake News Detector based on ML Algorithms. The issue of identifying fake news on social media has garnered significant interest in recent times. This project aims to use Machine learning algorithms to detect fake news directly based on the text content of news articles.
 This machine learning model is designed to predict whether a news article is fake or real based on its text content. The algorithm used in this model is the Passive Aggressive Classifier, which is a popular algorithm for online learning tasks.

The model is trained using the following steps:

Data Split: The provided dataset is split into training and testing sets using the train_test_split function from scikit-learn. The text column from the DataFrame df is used as the input feature, and the labels are the target variable. The training set comprises 80% of the data, while the testing set contains the remaining 20%.

Feature Extraction: The TfidfVectorizer from scikit-learn is used to convert the text data into numerical feature vectors. This vectorizer calculates the TF-IDF (Term Frequency-Inverse Document Frequency) values for each word in the text. It removes common English stop words and applies a maximum document frequency threshold of 0.7 to filter out infrequent terms. The vectorizer is fit on the training data using the fit_transform method, which learns the vocabulary and transforms the text into a numerical representation.

Model Training: The PassiveAggressiveClassifier from scikit-learn is initialized with a maximum of 50 iterations. This classifier is a linear model that is well-suited for online learning scenarios, where new data becomes available over time. It is trained on the transformed training data (tf_train) and the corresponding target labels (y_train) using the fit method.

Prediction: The trained model is used to make predictions on the testing data (tf_test) using the predict method. The predicted labels are stored in the y_pred variable.

Evaluation: The accuracy of the model is calculated by comparing the predicted labels (y_pred) with the true labels (y_test) using the accuracy_score function from scikit-learn. Additionally, a confusion matrix is computed using the confusion_matrix function, which provides a breakdown of the predicted labels and their actual values.

Model and Vectorizer Saving: Finally, both the trained model and the vectorizer are saved to disk using the pickle library. The model is saved as "finalized_model.pkl", while the vectorizer is saved as "vectorizer.pkl". These saved files can be loaded in the future for making predictions on new, unseen data.

To use this code for predicting fake and non-fake news, you need to have a dataset containing news articles with corresponding labels indicating whether they are fake or real. The dataset should be in a format similar to the provided DataFrame df, with a "text" column for the news content and a "labels" column for the target variable. You can adjust the test size and random state parameters in train_test_split to control the proportion of data used for testing and the randomness of the split.

The trained model can then be used to predict the authenticity of news articles by transforming the text data with the saved vectorizer (vector.transform()) and feeding it into the pac.predict() method. The model will output the predicted labels ('FAKE' or 'REAL').
<img width="948" alt="2" src="https://github.com/Srishti19-gaur/Veracity-Verse/assets/84332258/032de673-3a4f-4fe3-a246-b763d5b8d09f">
<img width="960" alt="1" src="https://github.com/Srishti19-gaur/Veracity-Verse/assets/84332258/53e53378-456e-481d-94b0-c4cc430590f4">
<img width="944" alt="3" src="https://github.com/Srishti19-gaur/Veracity-Verse/assets/84332258/16736821-fdf4-464e-9fd8-424ef8323b12">

<img width="947" alt="4" src="https://github.com/Srishti19-gaur/Veracity-Verse/assets/84332258/abb3cdf3-5385-4cc5-b014-5181252462ae">

