# Movie Genre Classification - Task 1

This project implements a machine learning model to predict the genre of a movie based on its plot description using the IMDb dataset.

## 📊 Project Overview
The goal is to classify movies into various genres using Natural Language Processing (NLP) and supervised learning techniques.

### Tech Stack
- **Languages:** Python
- **Libraries:** Pandas, NumPy, Seaborn, Matplotlib, Scikit-learn
- **Models Used:** LinearSVC, Multinomial Naive Bayes, Logistic Regression

---

## 🛠️ Data Visualization
We performed Exploratory Data Analysis (EDA) to understand the distribution of genres and description lengths.

* **Top Genres:** Identified the top 10 most frequent genres in the dataset.
* **Description Length:** Visualized the relationship between genre and the length of plot descriptions.

---

## 🚀 Model Performance
The models were trained using **TF-IDF Vectorization** with 100,000 features.

### Validation Results (LinearSVC):
- **Validation Accuracy:** ~58.36%
- **Key Metrics:** High precision and recall for popular genres like 'Drama' and 'Documentary'.

### Test Accuracy:
- **Test Accuracy:** 9.35% (Initial testing on solution data)

---

## 🔍 Predictor Function
I developed a custom function to predict genres for new, unseen movie descriptions.

**Example Usage:**
```python
def predict_movie(description):
    t_v1 = t_v.transform([description])
    pred_label = clf.predict(t_v1)
    return label_encoder.inverse_transform(pred_label)[0]

# Sample Tests:
# Input: "A movie where police cashes the criminal and shoot him" -> Output: Action
# Input: "A movie where person cashes a girl too get marry..." -> Output: Drama
