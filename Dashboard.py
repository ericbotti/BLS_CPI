import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")

def load_data():
    df = pd.read_csv('CPI_Data/cleaned_CPI_data.csv')
    return df

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Main app
def main():
    df = load_data() 

    local_css("styles.css")

    st.title('Structured Data Display')
    st.write("Here is the structured data with heatmap styling:")

    max_abs_value = df.iloc[:, 2:].abs().max().max()
    
    # Apply a color gradient with calculated vmin and vmax
    styled_df = df.style.format(formatter="{:.2%}", subset=df.columns[2:]).background_gradient(
        cmap='RdYlGn', subset=df.columns[2:], vmin=-max_abs_value, vmax=max_abs_value
    ).hide(axis="index")
    
    html = styled_df.to_html()

    # Display the styled DataFrame
    st.markdown(html, unsafe_allow_html=True)


if __name__ == "__main__":
    main()