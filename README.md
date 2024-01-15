# Overview
User behavior analysis, modelling and prediction by text classification and multimodal dataset from job ads dataset.

# Objective
- Objective 1: To predict the behavior of user based on the text description and location for multimodal model.
- Objective 2: To create a recommender system based on the text desciption and job classification.

# Dataset
- 2 sources of dataset
    - ads-50k-events.csv
    - ads-50k.json

# Play with Model
- 👉 [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://behaviorpredictionjobads.streamlit.app/)

# Training Results
|Model idx|Model name|Features col|Target col|Epochs|Accuracy (%)|
|---|---|---|---|---|---|
|  0|Naive-Bayes|title,abstract,content|kind|-|56.39   |
|  1|Bi-LSTM|title,abstract,content   |kind   |10   |56.88   |
|1.1|Bi-LSTM (multimodal)|title,abstract,content,location|kind|10   |57.39|
|1.1|Bi-LSTM (multimodal)|title,abstract,content,location|kind|20   |57.81|
|2|Bi-LSTM|title,abstract,content   |classification|10|8.50   |

## Analysis

- Deep learning modelling sucessfully exceed the accuracy of base model.
- Theoritically, by increasing the epochs of training, the model accuracy will improve.
- However, there is a limitation in terms of computation power due to model is trained by using Google Colab free GPU (Tesla T4). Kernel has down for some reasons.
- Possible further experimentation:
    - Case 1:
        - [ ] Training for 100 epochs
        - [ ] Using pre-trained embedding (USE) and Huggingface transformer model.
        - [ ] Using different columns for multi-modal model.
        - [ ] Using useful callbacks such as RecudeLROnPlateau and EarlyStopping.
    - Case 2:
        - [ ] Building similarity score algorithm based on embedding vector.


