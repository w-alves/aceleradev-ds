import streamlit as st
import pandas as pd


def main():
    st.title('Importing a csv and ploting with Pandas')
    file = st.file_uploader('Upload your file here')
    if file is not None:
        slider = st.slider('Range:', min_value=1, max_value=100)
        df = pd.read_csv(file)
        st.dataframe(df.head(slider))

if __name__ == '__main__':
        main()
