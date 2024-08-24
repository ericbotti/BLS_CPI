import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")

def load_data():
    NSA_MoM_df = pd.read_csv('CPI_Data/NSA_MoM_CPI_data.csv')
    NSA_YoY_df = pd.read_csv('CPI_Data/NSA_YoY_CPI_data.csv')
    SA_MoM_df = pd.read_csv('CPI_Data/SA_MoM_CPI_data.csv')
    SA_YoY_df = pd.read_csv('CPI_Data/SA_YoY_CPI_data.csv')
    return NSA_MoM_df, NSA_YoY_df, SA_MoM_df, SA_YoY_df

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

def display_dataframe(df):
    df.replace(np.nan, '', inplace=True)
    max_abs_value = df.iloc[0, 4:].abs().max().max()  # Adjust to select from the right columns
    styled_df = df.style.format(formatter="{:.2%}", subset=df.columns[4:]).background_gradient(
        cmap='RdYlGn', subset=df.columns[4:], vmin=-max_abs_value, vmax=max_abs_value
    ).apply(apply_styles, axis=1, subset=['Category', 'Sub Category 1', 'Sub Category 2']).hide(axis="index")
    html = styled_df.to_html()
    st.markdown(html, unsafe_allow_html=True)


# Main app
def main():
    NSA_MoM_df, NSA_YoY_df, SA_MoM_df, SA_YoY_df = load_data()

    local_css("styles.css")

    st.title('BLS CPI Data Analysis')
    st.write("Analysis of the Consumer Price Index (CPI) data from the Bureau of Labor Statistics (BLS).")

    SA = st.radio("Seasonality Adjustment:", ('Yes', 'No'))

    MoM_YoY = st.radio("Percentage change:", ('Month over Month', 'Year over Year'))   
    
    if SA == 'Yes' and MoM_YoY == 'Month over Month':
        display_dataframe(SA_MoM_df)
    elif SA == 'Yes' and MoM_YoY == 'Year over Year':
        display_dataframe(SA_YoY_df)
    elif SA == 'No' and MoM_YoY == 'Month over Month':
        display_dataframe(NSA_MoM_df)
    elif SA == 'No' and MoM_YoY == 'Year over Year':
        display_dataframe(NSA_YoY_df)


if __name__ == "__main__":
    main()