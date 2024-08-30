import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(layout="wide") # set Streamlit to wide mode to fit the entire screen

def load_data() -> tuple:
    '''
    Load the CPI data and summary text.
    
    Returns:
        A tuple of DataFrames and a string containing the summary text.
    '''
    NSA_MoM_df = pd.read_csv('CPI_Data/NSA_MoM_CPI_data.csv')
    NSA_YoY_df = pd.read_csv('CPI_Data/NSA_YoY_CPI_data.csv')
    SA_MoM_df = pd.read_csv('CPI_Data/SA_MoM_CPI_data.csv')
    SA_YoY_df = pd.read_csv('CPI_Data/SA_YoY_CPI_data.csv')
    with open('CPI_Data/summary.txt', 'r') as file:
        summary = file.read()
    return NSA_MoM_df, NSA_YoY_df, SA_MoM_df, SA_YoY_df, summary

def local_css(file_name: str):
    '''Load local CSS file to style the Streamlit app'''
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def apply_styles(row: pd.Series) -> list:
    '''
    Apply styles to the table based on the Category and Sub Category columns.

    Args:
        row: A row in the DataFrame

    Returns:
        A list of styles to apply to the row
    '''
    styles = [''] * len(row)
    # Unfortunately I couldn't find a way to add these styles in the CSS file, probably because of the way the table is rendered
    if row['Sub Category 1'] == '' and row['Sub Category 2'] == '':
        styles[0] = 'background-color: darkblue; color: white'
        styles[1] = 'background-color: darkblue; color: white'
        styles[2] = 'background-color: darkblue; color: white'
    elif row['Sub Category 1'] != '' and row['Sub Category 2'] == '':
        styles[1] = 'background-color: blue; color: white'
        styles[2] = 'background-color: blue; color: white'
    elif row['Sub Category 2'] != '':
        styles[2] = 'background-color: lightblue; color: black'
    return styles

def apply_index_weighting(df: pd.DataFrame):
    '''
    Calculate the percentage change in the data based on the weight of each category.

    Args:
        df: The DataFrame containing the data
    '''
    df.iloc[:, 4:] = df.iloc[:, 4:].apply(lambda x: x * df['Weight']/100, axis=0)


def display_dataframe(df: pd.DataFrame, index_weighted: str, num_columns_to_show: int):
    '''
    Display the DataFrame in a styled format, using a heatmap to highlight the values. The colors are based on the maximum absolute value in the first row. The reason for this is to have a consistent color scale across all rows, and the first row is chosen because it is the reference point for the index weighting.

    Args:
        df: The DataFrame to display
        index_weighted: A string indicating if the data is index weighted
        num_columns_to_show: The number of columns to display
    '''
    df.replace(np.nan, '', inplace=True)
    if index_weighted == 'Yes':
        apply_index_weighting(df)
    columns_to_show = ['Category', 'Sub Category 1', 'Sub Category 2', 'Weight'] + list(df.columns[4:4 + num_columns_to_show])
    df = df[columns_to_show]
    max_abs_value = df.iloc[0, 4:].abs().max().max() 
    df['Weight'] = df['Weight'].apply(lambda x: f"{x:.1f}%")

    table_styles = [ # still problems adding these styles to the CSS file
        {'selector': 'th:nth-child(1), td:nth-child(1)', 'props': 'width: 127px;'},
        {'selector': 'th:nth-child(2), td:nth-child(2)', 'props': 'width: 127px;'},
        {'selector': 'th:nth-child(3), td:nth-child(3)', 'props': 'width: 127px;'},
        {'selector': 'th:nth-child(4), td:nth-child(4)', 'props': 'width: 80px;'},
        {'selector': 'th:nth-child(n+5), td:nth-child(n+5)', 'props': 'min-width: 50px;'}
    ]
    # Apply heatmap to the data and apply styles to the table. The color palette has been chosen to be red-green to show negative and positive changes. This corresponds to the RdYlGn colormap in Matplotlib.
    styled_df = df.style.format(formatter="{:.2%}", subset=df.columns[4:]).background_gradient( # apply heatmap to the data. The 
        cmap='RdYlGn', subset=df.columns[4:], vmin=-max_abs_value, vmax=max_abs_value
    ).apply(apply_styles, axis=1, subset=['Category', 'Sub Category 1', 'Sub Category 2']).hide(axis="index")

    styled_df.set_table_styles(table_styles)

    html = styled_df.set_table_attributes('class="table-min-width"').to_html() # add a class to the table to apply the CSS styles in the CSS file
    st.markdown(html, unsafe_allow_html=True)

def prepare_plot_data(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Prepare the DataFrame for plotting by melting it to long format, which is easier to plot.

    Args:
        df: The DataFrame to prepare
    
    Returns:
        A DataFrame in long format
    '''
    # Melt the DataFrame to long format just for plotting
    plot_data = df.melt(id_vars=['Name', 'Category', 'Sub Category 1', 'Sub Category 2', 'Weight'],
                        var_name='Month-Year',
                        value_name='MoM_change')
    plot_data['MoM_change'] = plot_data['MoM_change'] * plot_data['Weight'] / 100
    plot_data['Month-Year'] = pd.to_datetime(plot_data['Month-Year'], format='%b-%y')
    plot_data.sort_values('Month-Year', inplace=True)
    return plot_data

def plot_data(df: pd.DataFrame, selected_ids: list):
    '''
    Plot the data based on the selected category.

    Args:
        df: The DataFrame to plot
        selected_ids: A list of selected categories to plot
    '''
    if not selected_ids:
        st.error("Please select at least one category to display.")
        return
    plot_df = prepare_plot_data(df)
    df_filtered = plot_df[plot_df['Name'].isin(selected_ids)]
    fig = px.line(df_filtered, x='Month-Year', y='MoM_change', color='Name', 
                  labels={'MoM_change': 'Month-over-Month Change (%)', 'Month-Year': 'Date'},
                  title='MoM Change Over Time')
    fig.update_layout(yaxis_tickformat='0.2%')
    st.plotly_chart(fig, use_container_width=True)

def main():
    NSA_MoM_df, NSA_YoY_df, SA_MoM_df, SA_YoY_df, summary = load_data()

    local_css("styles.css") # load the CSS file to style the Streamlit app

    st.title('BLS CPI Data Analysis')
    st.markdown('<textarea class="textbox">{}</textarea>'.format(summary), unsafe_allow_html=True) # display the summary text in a styled textarea

    tab1, tab2 = st.tabs(["Heatmap", "Plot"])
    with tab1:
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        with col1:
            num_columns_to_show = st.slider("Number of Periods to Display:", 1, 12, 6)
        with col2:
            index_weighted = st.radio("Index Weighted", ('Yes', 'No'), index=1)
        with col3:
            SA = st.radio("Seasonality Adjustment:", ('Yes', 'No'), key="seasonality_adjustment_tab1")
        with col4:
            MoM_YoY = st.radio("Percentage change:", ('Month over Month', 'Year over Year'))
        if SA == 'Yes' and MoM_YoY == 'Month over Month':
            display_dataframe(SA_MoM_df.iloc[:,1:], index_weighted, num_columns_to_show)
        elif SA == 'Yes' and MoM_YoY == 'Year over Year':
            display_dataframe(SA_YoY_df.iloc[:,1:], index_weighted, num_columns_to_show)
        elif SA == 'No' and MoM_YoY == 'Month over Month':
            display_dataframe(NSA_MoM_df.iloc[:,1:], index_weighted, num_columns_to_show)
        elif SA == 'No' and MoM_YoY == 'Year over Year':
            display_dataframe(NSA_YoY_df.iloc[:,1:], index_weighted, num_columns_to_show)

    with tab2:
        tab4, tab5 = st.columns([0.2, 1])
        with tab4:
            data_choice = st.radio("Seasonality Adjustment:", ('Yes', 'No'), key="seasonality_adjustment_tab2")
        df_to_use = SA_MoM_df if data_choice == 'Yes' else NSA_MoM_df
        all_ids = df_to_use['Name'].unique()
        default_selections = ['Services excluding Energy', 'Supercore'] # select default plots to display
        valid_defaults = [name for name in default_selections if name in all_ids] # check if default selections are present in the data
        with tab5:
            selected_ids = st.multiselect('Select IDs to plot:', all_ids, default=valid_defaults)
        plot_data(df_to_use, selected_ids)


if __name__ == "__main__":
    main()