import os
import requests
import json
import prettytable
import pandas as pd
from bs4 import BeautifulSoup
from transformers import BartForConditionalGeneration, BartTokenizer
from datetime import datetime

def initialize_cpi_data() -> tuple:
    '''
    Initializes the CPI data setup by creating the necessary folder for data storage, defining category weights, 
    mapping series names to their respective IDs, and preparing category mappings for organizing data. These have to be set manually because the BLS API doesn't provide this information.
    This function ensures that all necessary directories and configurations are in place for subsequent CPI data processing tasks.

    Returns:
        A tuple containing:
        - weight_map: A dictionary mapping CPI categories to their respective weights.
        - series_names: A dictionary mapping BLS series IDs to human-readable names.
        - series_ids: A list of BLS series IDs used for data retrieval.
        - folder_name: The name of the folder where CPI data will be stored.
        - category_map: A dictionary that maps category names to tuples containing their ordering and descriptive information.
    '''
    folder_name = 'CPI_Data' # folder directory containing all the saved data
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # CPI categories weights
    food_weight = 13.555
    energy_weight = 6.655
    services_less_energy_weight = 60.899
    shelter_weight = 36.191
    weight_map = {
        'All_Items': 100,
        'Food_Energy': food_weight + energy_weight,
        'Food': food_weight,
        'Food_At_Home': 8.167,
        'Food_Away_From_Home': 5.388,
        'Energy': energy_weight,
        'All_Items_Less_Food_Energy': 79.790,
        'Commodities_Less_Food_Energy_Commodities': 18.891,
        'Services_Less_Energy_Services': services_less_energy_weight,
        'Shelter': shelter_weight,
        'Supercore': services_less_energy_weight - shelter_weight 
    }

    # CPI id map
    series_names = {
        'CUUR0000SA0': 'NSA_All_Items',
        'CUSR0000SA0': 'SA_All_Items',
        'CUUR0000SA0L1E': 'NSA_All_Items_Less_Food_Energy',
        'CUSR0000SA0L1E': 'SA_All_Items_Less_Food_Energy',
        'CUUR0000SAF': 'NSA_Food',
        'CUSR0000SAF': 'SA_Food',
        'CUUR0000SAF11': 'NSA_Food_At_Home',
        'CUSR0000SAF11': 'SA_Food_At_Home',
        'CUUR0000SEFV': 'NSA_Food_Away_From_Home',
        'CUSR0000SEFV': 'SA_Food_Away_From_Home',
        'CUUR0000SA0E': 'NSA_Energy',
        'CUSR0000SA0E': 'SA_Energy',
        'CUUR0000SACL1E': 'NSA_Commodities_Less_Food_Energy_Commodities',
        'CUSR0000SACL1E': 'SA_Commodities_Less_Food_Energy_Commodities',
        'CUUR0000SASLE': 'NSA_Services_Less_Energy_Services',
        'CUSR0000SASLE': 'SA_Services_Less_Energy_Services',
        'CUUR0000SAH1': 'NSA_Shelter',
        'CUSR0000SAH1': 'SA_Shelter',
    }
    series_ids = list(series_names.keys())

    # Category map which maps the category and sub categories to their respective names
    category_map = {
        'All_Items': (0, 'Headline', '', '', 'All Items'),
        'Food_Energy': (1, 'Food + Energy', '', '', 'Food and Energy'),
        'Food': (2, '', 'Food', '', 'Food'),
        'Food_At_Home': (3, '', '', 'At home', 'Food at Home'),
        'Food_Away_From_Home': (4, '', '', 'Away Home', 'Food Away from Home'),
        'Energy': (5, '', 'Energy', '', 'Energy'),
        'All_Items_Less_Food_Energy': (6, 'Core', '', '', 'Core CPI'),
        'Commodities_Less_Food_Energy_Commodities': (7, '', 'Goods', '', 'Goods'),
        'Services_Less_Energy_Services': (8, '', 'Services', '', 'Services excluding Energy'),
        'Shelter': (9, '', '', 'Shelter', 'Shelter'),
        'Supercore': (10, '', '', 'Supercore', 'Supercore')
    }

    return weight_map, series_names, series_ids, folder_name, category_map

def retrieve_CPI_data(series_ids: list, series_names: dict, folder_name: str) -> pd.DataFrame:
    '''
    Retrieve the latest CPI data from the BLS website using the BLS API. The data is retrieved for the last 12 months and saved in a CSV file.

    Args:
        series_ids: A list of BLS series IDs to retrieve data for.
        series_names: A dictionary mapping BLS series IDs to human-readable names.
        folder_name: The name of the folder where the CPI data will be stored.

    Returns:
        A DataFrame containing the retrieved CPI data.
    '''
    current_date = datetime.now()
    current_year = current_date.year
    last_year = current_year - 1

    # API request structure taken from the BLS API documentation and modified to fit the data we need
    headers = {'Content-type': 'application/json'}
    data = json.dumps({"seriesid": series_ids, "startyear": str(last_year), "endyear": str(current_year)})
    response = requests.post('https://api.bls.gov/publicAPI/v2/timeseries/data/', data=data, headers=headers)
    json_data = json.loads(response.text)

    all_data = []
    for series in json_data['Results']['series']:
        rows = []
        for item in series['data']:
            footnotes = "".join([footnote['text'] + ',' for footnote in item['footnotes'] if footnote]).rstrip(',')
            if 'M01' <= item['period'] <= 'M12':
                rows.append([series_names[series['seriesID']], item['year'], item['period'], item['value'], footnotes])

        df = pd.DataFrame(rows, columns=["series id", "year", "period", "value", "footnotes"])
        all_data.append(df)

    complete_data = pd.concat(all_data)

    # Process the data to properly save them in a csv file
    df = complete_data.copy()
    df['date'] = pd.to_datetime(df['year'].astype(str) + df['period'].str.replace('M', ''), format='%Y%m')
    df['series id'] = df['series id'].astype(str) 
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df['footnotes'] = df['footnotes'].astype(str) 
    df.drop(['year', 'period', 'footnotes'], axis=1, inplace=True)
    df.rename(columns={'series id': 'id'}, inplace=True)
    df = df[['id', 'date', 'value']]

    csv_path = os.path.join(folder_name, 'CPI_data.csv')
    df.to_csv(csv_path, index=False)
    return df

def MoM_YoY_CPI_data(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Calculate the Month-over-Month (MoM) and Year-over-Year (YoY) changes in the CPI data.

    Args:
        df: The DataFrame containing the CPI data.

    Returns:
        The DataFrame with the MoM and YoY changes added.
    '''
    df['MoM_change'] = df.groupby('id')['value'].transform(lambda x: (x - x.shift(-1)) / x.shift(-1))
    df['YoY_change'] = df.groupby('id')['value'].transform(lambda x: (x - x.shift(-12)) / x.shift(-12))
    return df

def calculate_weighted_change(df: pd.DataFrame, id1: str, id2: str, new_id: str, weight1: float, weight2: float, weight_total: float, operation: str) -> pd.DataFrame:
    '''
    Calculate the weighted change of food + energy and supercore for both SA and non-SA (NSA). The supercore CPI corresponds to the core-core CPI, which excludes shelter and energy. The weighted change is calculated by adding or subtracting the weighted changes of the two series.
    
    Args:
        df: The DataFrame containing the CPI data.
        id1: The id of the first series.
        id2: The id of the second series.
        new_id: The id of the new series.
        weight1: The weight of the first series.
        weight2: The weight of the second series.
        weight_total: The total weight of the new series.
        operation: The operation to be performed. Must be 'add' or 'subtract'.
    
    Returns:
        The DataFrame containing the new series
    '''
    series1 = df.loc[df['id'] == id1, ['MoM_change', 'YoY_change']].set_index(df.loc[df['id'] == id1, 'date']) * weight1
    series2 = df.loc[df['id'] == id2, ['MoM_change', 'YoY_change']].set_index(df.loc[df['id'] == id2, 'date']) * weight2

    if operation == 'add':
        result = (series1 + series2) / weight_total
    elif operation == 'subtract':
        result = (series1 - series2) / weight_total
    else:
        raise ValueError("Operation must be 'add' or 'subtract'.")

    result = result.reset_index()
    result['id'] = new_id
    result['value'] = 0
    result = result[['id', 'date', 'value', 'MoM_change', 'YoY_change']]
    return result

def process_cpi_data(df: pd.DataFrame, category_map: dict, weight_map: dict) -> tuple:
    '''
    Create four DataFrames containing processed data for NSA MoM, NSA YoY, SA MoM, and SA YoY. The data is organized by category and subcategory, and the values are pivoted to show the changes over time. This allows the data to be easily visualized later in a Streamlit dashboard.

    Args:
        df: The DataFrame containing the CPI data.
        category_map: A mapping from category IDs to category names and metadata.
        weight_map: A mapping from category IDs to their respective weights.

    Returns:
        Four DataFrames containing processed data for NSA MoM, NSA YoY, SA MoM, and SA YoY.
    '''
    ordered_categories = ['Headline', 'Food + Energy', 'Core']
    ordered_sub_categories_1 = ['Food', 'Energy', 'Commodities', 'Services']
    ordered_sub_categories_2 = ['At home', 'Away Home', 'Shelter', 'Supercore']

    NSA_MoM_df, NSA_YoY_df, SA_MoM_df, SA_YoY_df = None, None, None, None

    for i in range(4):
        if i in [0, 1]:
            data = df[df['id'].str.startswith('NSA_')].copy()
            prefix_length = 4
        else:
            data = df[df['id'].str.startswith('SA_')].copy()
            prefix_length = 3

        data.loc[:, 'id'] = data['id'].str[prefix_length:]
        data.loc[:, 'Month-Year'] = data['date'].dt.strftime('%b-%y')
        order_cat = data['id'].apply(lambda x: category_map.get(x, (None, None, None, None, None)))
        data.loc[:, 'Order'] = [item[0] for item in order_cat]
        data.loc[:, 'Category'] = [item[1] for item in order_cat]
        data.loc[:, 'Sub Category 1'] = [item[2] for item in order_cat]
        data.loc[:, 'Sub Category 2'] = [item[3] for item in order_cat]
        data.loc[:, 'Name'] = [item[4] for item in order_cat]
        data.loc[:, 'Weight'] = data['id'].map(weight_map).fillna('Unknown')

        if i in [0, 2]:
            value_column = 'MoM_change'
        else:
            value_column = 'YoY_change'

        data_pivot = data.pivot_table(
            index=['Name', 'Order', 'Category', 'Sub Category 1', 'Sub Category 2', 'Weight'],
            columns='Month-Year',
            values=value_column,
            aggfunc='first'
        )

        data_pivot = data_pivot[sorted(data_pivot.columns, key=lambda x: pd.to_datetime(x, format='%b-%y'), reverse=True)]
        data_pivot.columns.name = None
        data_pivot.reset_index(inplace=True)
        data_pivot.sort_values(by='Order', inplace=True)
        data_pivot.drop(columns=['Order'], inplace=True)

        if i == 0:
            NSA_MoM_df = data_pivot
        elif i == 1:
            NSA_YoY_df = data_pivot
        elif i == 2:
            SA_MoM_df = data_pivot
        else:
            SA_YoY_df = data_pivot

    return NSA_MoM_df, NSA_YoY_df, SA_MoM_df, SA_YoY_df

def save_cpi_dataframes_to_csv(NSA_MoM_df: pd.DataFrame, NSA_YoY_df: pd.DataFrame, SA_MoM_df: pd.DataFrame, SA_YoY_df: pd.DataFrame, folder_name: str):
    """
    Saves the CPI DataFrames to CSV files in the specified folder.

    Args:
        NSA_MoM_df: DataFrame containing NSA Month-over-Month CPI data.
        NSA_YoY_df: DataFrame containing NSA Year-over-Year CPI data.
        SA_MoM_df: DataFrame containing SA Month-over-Month CPI data.
        SA_YoY_df: DataFrame containing SA Year-over-Year CPI data.
        folder_name: The directory where the CSV files will be saved. Defaults to 'CPI_Data'.
    """
    NSA_MoM_path = os.path.join(folder_name, 'NSA_MoM_CPI_data.csv')
    NSA_YoY_path = os.path.join(folder_name, 'NSA_YoY_CPI_data.csv')
    SA_MoM_path = os.path.join(folder_name, 'SA_MoM_CPI_data.csv')
    SA_YoY_path = os.path.join(folder_name, 'SA_YoY_CPI_data.csv')

    NSA_MoM_df.to_csv(NSA_MoM_path, index=False)
    NSA_YoY_df.to_csv(NSA_YoY_path, index=False)
    SA_MoM_df.to_csv(SA_MoM_path, index=False)
    SA_YoY_df.to_csv(SA_YoY_path, index=False)

def fetch_report_text(url: str) -> str:
    '''
    The BLS API doesn't provide the report text, so I had to scrape the website to get it. The website structure is simple, so I used BeautifulSoup to get the text. BLS has an anti-scraping system, so I had to add a User-Agent header to the request.

    Args:
        url: The URL of the webpage to scrape
    
    Returns:
        The text of the report
    '''
    headers = {
        'User-Agent': 'email@domain.name'  # it doesn't matter
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200: # code 200 means the request was successful
        soup = BeautifulSoup(response.text, 'html.parser')
        normalnews_div = soup.find('div', class_='normalnews') # after analyzing the webpage using the inspector, I found that the needed text is inside a <div> with class 'normalnews'
        if normalnews_div:
            return normalnews_div.get_text(separator='\n', strip=True)
        else:
            return "No <div> with class 'normalnews' found." # error management
    else:
        return f"Failed to retrieve page: {response.status_code}"
    
def extract_paragraph_based_on_index(full_text: str) -> str:
    '''
    Extract the second paragraph of the main section of the report, or the third paragraph if a "NOTE:" is present after the second paragraph (e.g. June 2024). A LLM could be used to find the paragraph of interest, but after the whole process of training, the best result would be for it to understand this pattern. Therefore, I decided to use a simple rule-based approach. For all the previous year, this rule-based approach worked perfectly. A more advanced system that could detect a sudden change in the report structure would be needed to handle the edge cases, but for this project would be overkill.

    Args:
        full_text: The full text of the report retrieved from the BLS website

    Returns:
        The paragraph of interest
    '''
    paragraphs = [p.strip() for p in full_text.replace('\r', '').split('\n\n') if p.strip()] # split the text into paragraphs
    for i, paragraph in enumerate(paragraphs):
        if "CONSUMER PRICE INDEX" in paragraph: # "CONSUMER PRICE INDEX" is the section we are interested in, and it's the only consumer price index title written in uppercase
            start_index = i
            break
    else:
        return "CONSUMER PRICE INDEX section not found."
    note_present = "NOTE:" in paragraphs[start_index + 1] # check if a "NOTE:" is present immediately after "CONSUMER PRICE INDEX"
    target_index = start_index + 3 if note_present else start_index + 2
    if target_index < len(paragraphs):
        target_paragraph = paragraphs[target_index].replace('\n', ' ') # replace newline characters with a space
        return target_paragraph
    else:
        return "The paragraph of interest could not be found."
    

def summarize_paragraph(paragraph: str) -> str:
    """
    I decided to use a BART pre-trained model to summarize the paragraph retrieved above, and ignore the food section of it. The model is fine-tuned for summarization tasks, and it should be able to generate a good summary of the paragraph. The model is not perfect, and the summary could be better, but it should be good enough for this project. I already used BART previously, so I know the capabilities of the model. The only slight issue is that the paragraph is not very long, and the model doesn't have a lot of margin to generate a good summary.

    Args:
        paragraph (str): The paragraph to summarize.

    Returns:
        str: The summarized text.
    """
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    input_text = "Summarize and ignore the food parts: " + paragraph # prompt for the model. This could be improved by adding more information to the prompt, but for this project, it should be enough.
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    # # The following parameters have been fine-tuned with the past 10 months of CPI reports
    summary_ids = model.generate(
        input_ids,
        num_beams=4,
        min_length=20,
        max_length=60,
        length_penalty=1.0, # no penalty here works better because the paragraph is short
        no_repeat_ngram_size=3,  # prevent repetition of phrases
        early_stopping=True  # for short paragraphs like this one it's not really needed
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return summary


def main():
    weight_map, series_names, series_ids, folder_name, category_map = initialize_cpi_data()

    df = retrieve_CPI_data(series_ids, series_names, folder_name)
    print("CPI data retrieved.")
    df = MoM_YoY_CPI_data(df)

    # Calculate 'SA_Food_Energy' and 'NSA_Food_Energy'
    df = pd.concat([
        df,
        calculate_weighted_change(df, 'SA_Food', 'SA_Energy', 'SA_Food_Energy', weight_map['Food'], weight_map['Energy'], weight_map['Food_Energy'], 'add'),
        calculate_weighted_change(df, 'NSA_Food', 'NSA_Energy', 'NSA_Food_Energy', weight_map['Food'], weight_map['Energy'], weight_map['Food_Energy'], 'add')
    ])

    # Calculate 'SA_Supercore' and 'NSA_Supercore'
    df = pd.concat([
        df,
        calculate_weighted_change(df, 'SA_Services_Less_Energy_Services', 'SA_Shelter', 'SA_Supercore', weight_map['Services_Less_Energy_Services'], weight_map['Shelter'], weight_map['Supercore'], 'subtract'),
        calculate_weighted_change(df, 'NSA_Services_Less_Energy_Services', 'NSA_Shelter', 'NSA_Supercore', weight_map['Services_Less_Energy_Services'], weight_map['Shelter'], weight_map['Supercore'], 'subtract')
    ])

    NSA_MoM_df, NSA_YoY_df, SA_MoM_df, SA_YoY_df = process_cpi_data(df, category_map, weight_map)
    save_cpi_dataframes_to_csv(NSA_MoM_df, NSA_YoY_df, SA_MoM_df, SA_YoY_df, folder_name)
    print("CPI data processed and saved.")

    url = "https://www.bls.gov/news.release/cpi.nr0.htm" # URL of BLS CPI latest report
    full_text = fetch_report_text(url)

    relevant_paragraph = extract_paragraph_based_on_index(full_text)

    summary = summarize_paragraph(relevant_paragraph)

    txt_path = os.path.join(folder_name, 'summary.txt')
    with open(txt_path, 'w') as file:
        file.write(summary)
    print("CPI report summary generated.")

if __name__ == "__main__":
    main()