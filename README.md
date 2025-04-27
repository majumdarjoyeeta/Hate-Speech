# Hate-Speech
# 🗣️ Hate Speech Detection using LSTM 🚀
🔗 **GitHub Repository**: [majumdarjoyeeta/Hate-Speech](https://github.com/majumdarjoyeeta/Hate-Speech)

## 📌 Project Overview
This project aims to detect **Hate Speech**, **Offensive Language**, and **Neutral** comments using **LSTM (Long Short Term Memory)** networks in **Deep Learning** 🧠. It leverages **Natural Language Processing (NLP)** techniques to process and classify tweets into three categories:

- **0**: Hate Speech 🛑
- **1**: Offensive Language 😡
- **2**: Neither 🟢

Using **LSTM** combined with **spaCy** and **SMOTE** (Synthetic Minority Over-sampling Technique) for class balancing, this model can classify text data effectively. The **model accuracy** and **confusion matrix** provide insights into the classification performance 🎯.

---

## ⚙️ Installation
To run this project, install the necessary libraries using **pip**:

```bash
pip install pandas numpy spacy tensorflow imbalanced-learn matplotlib seaborn
python -m spacy download en_core_web_sm
📚 Dataset
The dataset used for this project is available on Kaggle 🌐:

Hate Speech and Offensive Language Detection Dataset:
Dataset Link

The dataset contains tweets labeled as hate speech, offensive language, or neutral.

🔍 Project Workflow
1. Data Preprocessing 🧹
Remove unwanted columns ✂️.

Check for missing values 🛠️.

Clean text by removing non-alphabetic characters and extra whitespaces 🧼.

Lemmatization using spaCy to convert words to their base form 🔄.

Stopwords removal to eliminate common words like "the", "and", etc. ❌

2. Text Encoding 🔡
Text is one-hot encoded for input into the LSTM model 💡.

Sequences are padded to ensure they have a consistent length.

3. Class Balancing ⚖️
SMOTE is used to oversample the minority classes, ensuring a balanced dataset 👥.

4. Model Architecture 🏗️
We define an LSTM model with the following layers:

Embedding Layer 🧩 to represent words as vectors.

LSTM Layers 📊 to capture the sequence dependencies.

Dropout Layer 💧 to prevent overfitting.

Dense Layer 🎯 for classification into 3 categories (Hate, Offensive, Neutral).

python
Copy
Edit
model = models.Sequential([
    layers.Embedding(vocab_size, embedding_dim, input_length=max_len),
    layers.LSTM(100, return_sequences=True),
    layers.Dropout(0.3),
    layers.LSTM(50, return_sequences=True),
    layers.LSTM(50),
    layers.Dense(3, activation='softmax')
])
5. Training 🚀
The model is trained on the balanced data for 10 epochs with a batch size of 32. The model uses the Adam optimizer for efficient learning.

6. Evaluation 📝
After training, the model’s performance is evaluated on the test data, and the accuracy is printed 📊.

7. Classification Report & Confusion Matrix 📊
The classification report gives detailed precision, recall, and F1 scores, while the confusion matrix visualizes the model’s predictions.

python
Copy
Edit
# Confusion Matrix
plt.figure(figsize=(8,6))
cm = confusion_matrix(y_test, y_pred, normalize='true')
sns.heatmap(cm, annot=True, cmap='Blues', fmt=".2f", xticklabels=["Hate", "Offensive", "Neutral"], yticklabels=["Hate", "Offensive", "Neutral"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Normalized Confusion Matrix")
plt.tight_layout()
plt.show()
📈 Results
Accuracy 🏆: The model performs well in classifying tweets into hate speech, offensive language, or neutral categories.

Confusion Matrix 🧮: Visual representation of the true vs predicted labels.

🎯 What's Next? 🚀
Here are some potential improvements to enhance the model further:

Fine-tune the model with additional data 📦.

Hyperparameter tuning to improve accuracy 📈.

Deploy the model using Flask or FastAPI for real-time predictions 🌍.

User Interface using Gradio or Streamlit for easy input and results 📱.

🤝 Contribution
Feel free to fork 🍴, star ⭐, or open issues!
Pull Requests (PRs) to improve model performance, add new features, or contribute to the dataset are welcome! 🎉

👩‍💻 Author
Made with ❤️ by Joyeeta Majumdar
🔗 Visit the GitHub Profile

Let's make the internet a safer place by detecting hate speech effectively! 🌐🛑

📁 Repository
Check out the full project here:
👉 Hate Speech Detection using LSTM

yaml
Copy
Edit

---

### Key Features:
- **Emojis** are used extensively to enhance visual appeal and make it fun to read! 🎉
- **Detailed workflow** explaining the steps from data preprocessing to model evaluation.
- **Clear explanations** for each section, making it easy for others to understand how to implement the solution and contribute to it.



