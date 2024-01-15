import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

project_path = Path("./behavior_prediction_jobads_text")
model_name = "model_1_1_multimodal_epoch_20"
ohe_location_train_load = np.load("onehotencoder_location_train.npy")


def load_model():
    model = tf.keras.models.load_model(project_path / model_name)
    return model


def load_target_ohe():
    ohe = np.load(project_path / "target_ohe.npy")
    return ohe


def predict(input):
    model = load_model()

    model_preds = model.predict(input)
    return model_preds


def process_prediction(model_prediction):
    model_argmax = tf.math.argmax(tf.squeeze(model_prediction).numpy())
    model_pred_arg = load_target_ohe().categories_[0][model_argmax]
    model_preds_probs = tf.squeeze(model_prediction)[model_argmax]
    return model_pred_arg, model_preds_probs


def input_pipeline(job_description: str, location: str):
    data = {"text": [job_description], "location": [location]}
    input_df = pd.DataFrame(data)

    text = input_df["text"].to_numpy()
    location = input_df["location"].to_numpy().reshape(-1, 1)
    # print(text)
    # print(text.shape)
    # ohe_location = OneHotEncoder(sparse_output=False)
    try:
        location_data_ohe = ohe_location_train_load.transform(location)

        # print(location_data_ohe)
        # print(location_data_ohe.shape)
        # print(tf.squeeze(location_data_ohe).shape)
        # location_test_data_ohe = tf.one_hot(location_data_ohe, depth=50)
        input_slice = tf.data.Dataset.from_tensor_slices((text, location_data_ohe))

        # filler
        output_filler = tf.tile(tf.expand_dims(tf.zeros(2), axis=1), [1, 2])
        output_filler = tf.data.Dataset.from_tensor_slices(output_filler)

        input_zip = tf.data.Dataset.zip((input_slice, output_filler))

        user_input = input_zip.batch(1024).prefetch(tf.data.AUTOTUNE)

        return user_input
    except:
        print(f"ERROR: Location is {location} not found")
