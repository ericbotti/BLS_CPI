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

def apply_styles(row):
    styles = [''] * len(row)
    if row['Sub Category 1'] == '' and row['Sub Category 2'] == '':
        # Apply dark blue to Category, Sub Category 1, and Sub Category 2
        styles[0] = 'background-color: darkblue; color: white'
        styles[1] = 'background-color: darkblue; color: white'
        styles[2] = 'background-color: darkblue; color: white'
    elif row['Sub Category 1'] != '' and row['Sub Category 2'] == '':
        # Apply blue to Sub Category 1 and Sub Category 2
        styles[1] = 'background-color: blue; color: white'
        styles[2] = 'background-color: blue; color: white'
    elif row['Sub Category 2'] != '':
        # Apply light blue to Sub Category 2
        styles[2] = 'background-color: lightblue; color: black'
    return styles

# Main app
def main():
    df = load_data() 

    local_css("styles.css")

    st.title('Structured Data Display')
    st.write("Here is the structured data with heatmap styling:")

    df.replace(np.nan, '', inplace=True)

    max_abs_value = df.iloc[0, 4:].abs().max().max()
    
    # Apply a color gradient with calculated vmin and vmax
    styled_df = df.style.format(formatter="{:.2%}", subset=df.columns[4:]).background_gradient(
        cmap='RdYlGn', subset=df.columns[4:], vmin=-max_abs_value, vmax=max_abs_value
    ).apply(apply_styles, axis=1, subset=['Category', 'Sub Category 1', 'Sub Category 2']).hide(axis="index")
    
    html = styled_df.to_html()

    # Display the styled DataFrame
    st.markdown(html, unsafe_allow_html=True)


if __name__ == "__main__":
    main()