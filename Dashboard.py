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

def apply_index_weighting(df):
    df.iloc[:, 4:] = df.iloc[:, 4:].apply(lambda x: x * df['Weight']/100, axis=0)


def display_dataframe(df, index_weighted, num_columns_to_show):
    df.replace(np.nan, '', inplace=True)
    if index_weighted:
        apply_index_weighting(df)
    columns_to_show = ['Category', 'Sub Category 1', 'Sub Category 2', 'Weight'] + list(df.columns[4:4 + num_columns_to_show])
    df = df[columns_to_show]
    max_abs_value = df.iloc[0, 4:].abs().max().max() 
    df['Weight'] = df['Weight'].apply(lambda x: f"{x:.1f}%")

    table_styles = [
        {'selector': 'th:nth-child(1), td:nth-child(1)', 'props': 'width: 127px;'},
        {'selector': 'th:nth-child(2), td:nth-child(2)', 'props': 'width: 127px;'},
        {'selector': 'th:nth-child(3), td:nth-child(3)', 'props': 'width: 127px;'},
        {'selector': 'th:nth-child(4), td:nth-child(4)', 'props': 'width: 80px;'},
        {'selector': 'th:nth-child(n+5), td:nth-child(n+5)', 'props': 'min-width: 50px;'}
    ]

    styled_df = df.style.format(formatter="{:.2%}", subset=df.columns[4:]).background_gradient(
        cmap='RdYlGn', subset=df.columns[4:], vmin=-max_abs_value, vmax=max_abs_value
    ).apply(apply_styles, axis=1, subset=['Category', 'Sub Category 1', 'Sub Category 2']).hide(axis="index")

    styled_df.set_table_styles(table_styles)

    html = styled_df.set_table_attributes('class="table-min-width"').to_html()
    st.markdown(html, unsafe_allow_html=True)


# Main app
def main():
    NSA_MoM_df, NSA_YoY_df, SA_MoM_df, SA_YoY_df = load_data()

    local_css("styles.css")

    st.title('BLS CPI Data Analysis')
    st.write("Analysis of the Consumer Price Index (CPI) data from the Bureau of Labor Statistics (BLS).")

    index_weighted = st.checkbox("Index Weighted", value=False) 
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        SA = st.radio("Seasonality Adjustment:", ('Yes', 'No'))
    with col3:
        MoM_YoY = st.radio("Percentage change:", ('Month over Month', 'Year over Year'))
    with col1:
        if MoM_YoY == 'Month over Month':
            num_columns_to_show = st.slider("Number of Previous Months:", min_value=1, max_value=12, value=6)
        elif MoM_YoY == 'Year over Year':
            num_columns_to_show = st.slider("Number of Previous Years:", min_value=1, max_value=6, value=6)    
    
    if SA == 'Yes' and MoM_YoY == 'Month over Month':
        display_dataframe(SA_MoM_df, index_weighted, num_columns_to_show)
    elif SA == 'Yes' and MoM_YoY == 'Year over Year':
        display_dataframe(SA_YoY_df, index_weighted, num_columns_to_show)
    elif SA == 'No' and MoM_YoY == 'Month over Month':
        display_dataframe(NSA_MoM_df, index_weighted, num_columns_to_show)
    elif SA == 'No' and MoM_YoY == 'Year over Year':
        display_dataframe(NSA_YoY_df, index_weighted, num_columns_to_show)


if __name__ == "__main__":
    main()