import streamlit as st
import pandas as pd
import joblib

# THIS loads the trained model
model = joblib.load("profit_model.pkl")

st.title("Business Profit / Loss Predictor / MADE By Sumit Singh")
st.write("👆 Please upload a CSV file to start prediction")


file = st.file_uploader("Upload Your CSV File Here:", type=["csv"])

if file is not None:
    dataframe = pd.read_csv(file)
    required_columns = ["Sales", "Quantity", "Discount"]

    if not all(col in dataframe.columns for col in required_columns):

         st.error("❌ CSV must contain columns: Sales, Quantity, Discount")
    else:
        #input
        x = dataframe[["Sales", "Quantity", "Discount"]]
        #output
        dataframe["predicted_profit"] = model.predict(x)
        #repalcing the result to pretected profit
        dataframe["result"] = dataframe["predicted_profit"].apply(
         lambda x: "profit" if x>0 else "loss"    

        )

        st.dataframe(dataframe.head())
     #python -m streamlit run newfile.py command that need to paste in cmd to run the app
        