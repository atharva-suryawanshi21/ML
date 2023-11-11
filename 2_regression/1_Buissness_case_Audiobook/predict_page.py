import streamlit as st
import numpy as np
import tensorflow as tf


def load_model():
    loaded_model = tf.keras.models.load_model('saved_model.h5')
    return loaded_model


our_model = load_model()


def show_predict_page():

    st.title("Audiobook Customer Conversion Predictor")

    st.write("""### Give Useful Information""")
    data = [0.0] * 10
    data[0] = st.number_input("Enter overall book length:")
    data[1] = st.number_input("Enter average book length:")

    data[2] = st.number_input("Enter an overall price:")
    data[3] = st.number_input("Enter book price average:")

    data[4] = st.number_input("Enter '1' if submitted a review else 0:")
    data[5] = st.number_input("Enter review out of 10 : ")
    data[6] = st.number_input("Enter minutes listened:")
    data[7] = st.number_input("Enter completed book length:")
    data[8] = st.number_input("Enter number of support requests:")
    data[9] = st.number_input(
        "Enter the value of last visited minus purchase date:")

    ok = st.button("predict conversion")
    if ok:
        loaded_data = np.loadtxt('mean_and_std.txt')

        mean_values = loaded_data[:, 0]
        std_deviation_values = loaded_data[:, 1]

        for i in range(0, 9):
            data[i] = (data[i] - mean_values[i])/std_deviation_values[i]

        input_data = np.array(data)
        input_data = input_data.reshape((1, -1))
        conversion = our_model.predict(input_data)
        predicted_class = np.argmax(conversion, axis=1)

        if predicted_class == 0:
            st.subheader(
                " There is low probablity that the customer will back")
        else:
            st.subheader(
                "There is high probablity that the customer will back")
