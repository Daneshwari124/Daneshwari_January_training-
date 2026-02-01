# Assignment 04 – Spam Email Detection using SVM

## Dataset
The SMS Spam Collection dataset was taken from Kaggle.  
It contains 5574 messages labeled as spam or ham (not spam).

Target Variable: label (spam or not spam)

---

## Text Preprocessing
- Converted text messages into numerical features using TF-IDF
- Removed stopwords
- Converted labels into binary values

---

## Model
- Support Vector Machine (SVM) with linear kernel
- Trained on 80% of data and tested on 20%

---

## Model Evaluation
- Accuracy metric was used
- Classification report and confusion matrix were generated

---

## Results & Conclusion
- The SVM model achieved high accuracy in detecting spam messages
- TF-IDF representation helped capture important words
- SVM performed well for text classification problems
- The model successfully predicted spam for new unseen messages
