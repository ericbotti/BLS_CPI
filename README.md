# BLS CPI Assignment

## Introduction

This project creates a web app that displays key US CPI data using a heatmap, a plot, and a summary of the BLS CPI report. It shows month-over-month (MoM) and year-over-year (YoY) changes for the main CPI categories, with options for seasonally adjusted or unadjusted data, and data adjusted or not adjusted for category weights.

When the user runs the `script.py` file, the algorithm retrieves the current CPI data, processes it, and launches a Streamlit dashboard. The script remains active and, at 08:30 EST, it fetches the latest CPI data and updates the dashboard. By starting the script before 08:30 EST, users can view the latest CPI data as soon as it is released by the BLS.

To properly render the README.md file and read the documentation in VSCode, use Ctrl+Shift+V or Cmd+Shift+V (on Mac). The project uses Poetry to manage the versions of various libraries without interfering with locally installed versions. Some packages require older versions to work properly, so using Poetry is strongly recommended. A brief guide on how to use Poetry is provided at the end of this document.

## Structure

The project is divided into two scripts: `CPI.py` and `Dashboard.py`. These are called from the main script, `script.py`.

### CPI.py
This script is responsible for gathering CPI data from the BLS website using their [API](https://www.bls.gov/developers/api_python.htm) and process them, calculating the statistics of interest and generate the summary of the CPI report. The initial work was done in `CPI.ipynb`, which was subsequently transformed into a Python script. The `initialize_CPI_data` function contains the dictionaries with all the information needed for both the API and the script to retrieve just the CPI categories that will be shown in the Streamlit dashboard. The IDs can be found [here](https://data.bls.gov/dataQuery/find?fq=survey:[cu]&s=popularity:D&r=100&st=0). `retrieve_CPI_data` makes use of the BLS API to retrieve the needed data, using the example code provided by the [BLS website](https://www.bls.gov/developers/api_python.htm#python2). I couldn't find the 'food+energy' component of the Core CPI and the [Supercore CPI](https://www.stlouisfed.org/on-the-economy/2024/may/measuring-inflation-headline-core-supercore-services), therefore I used the [BLS guide](https://www.bls.gov/cpi/factsheets/constructing-special-cpis.htm) to calculate them using the other data. 

Regarding the CPI report summary, the BLS API doesn't provide the [current CPI report](https://www.bls.gov/news.release/cpi.nr0.htm), so web scraping is needed. It's important to note that the BLS website has an anti-scraping system in place, which prevents most of the web scraping systems to work properly. The workaround here is to use an User-agent header, rather than a browser agent (which BLS somehow still recognizes as a non-browser entity). This means adding a general email in the request header. The `extract_paragraph_based_on_index` function extracts the second paragraph of the main section of the report, or the third paragraph if a "NOTE:" is present after the second paragraph (e.g. June 2024). A LLM could be used to find the paragraph of interest, but after the whole process of training, the best result would be for it to understand this pattern. Therefore, I decided to use a simple rule-based approach. For all the previous years, this rule-based approach worked perfectly. A more advanced system that could detect a sudden change in the report structure would be needed to handle the edge cases, but for this project would be overkill. The `summarize_paragraph` then uses a [BART](https://huggingface.co/facebook/bart-large-cnn) pre-trained model to summarize the paragraph, and ignore the food section of it. The model is fine-tuned for summarization tasks, and it should be able to generate a good summary of the paragraph. Important is to notice that the paragraph is already very short, and every part of the sentences are important, therefore the summary is basically just the part of the CPI report which is of interest, without a strong summarization. The model is not perfect, and the summary could be better, but it should be good enough for this project. I already used BART previously, so I know the capabilities of the model. The only slight issue is that the paragraph is not very long, and the model doesn't have a lot of margin to generate a good summary.

### Dashboard.py
The dashboard is built using Streamlit, and the design modelled with a CSS style sheet (`styles.css`). It is mainly divided into generating a heat map using the MoM and YoY changes of the data gathered in `CPI.py`, and plotting the various CPI categories' MoM changes over time. The user can choose to display the number of months (default is 6), the data multiplied by the category weight (index weighted), the seasonally adjusted or not adjusted data, and MoM or YoY changes. The colors of the heatmap are based on the maximum absolute value in the first row. The reason for this is to have a consistent color scale across all rows, with the first row chosen as the reference point for the index weighting. The plotting system shows multiple CPI categories together (default: services excluding energy and supercore).


## Improvements

Several improvements can be implemented. First, the script retrieves the data every day at 08:30 EST but doesn't check if it is the CPI release date. Additionally, if the BLS changes the structure of the CPI reports, the script might extract the wrong paragraph, making the summary useless. An AI model could be implemented to recognize the paragraph of interest or detect changes in the report structure. This would ensure consistency and robustness in the CPI report summary generation. The BART model could also be further fine-tuned, or an in-house model could be built specifically for this task, but this would require significant time.


## Appendix: Poetry

Python must be already installed locally.
Then:
1. Install Poetry: `pip install poetry`
2. Access the project folder: `cd <project-directory>`
3. Install all the specific versions of the packages: `poetry install`
4. Activate the virtual environment: `poetry shell`
5. Run the script `python3 script.py`

After having done this the first time, then every time the user just want to use the script, should:
1. `cd <project-directory>`
2. `poetry shell`
3. `python3 script.py`