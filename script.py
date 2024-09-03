import schedule
import time
import subprocess
from datetime import datetime
import pytz

def run_cpi_script():
    '''
    Run the CPI data retrieval and processing script.
    '''
    print(f"Running CPI data retrieval script at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    subprocess.run(["python", "CPI.py"]) # subprocess.run() is used to run the script in the same process
    print("CPI data updated.")

def run_streamlit_app() -> subprocess.Popen:
    '''
    Function to run the Streamlit app.
    '''
    print("Starting the Streamlit app...")
    return subprocess.Popen(["streamlit", "run", "Dashboard.py"]) # subprocess.Popen() is used to run the Streamlit app in a separate process

def restart_streamlit_app(streamlit_process: subprocess.Popen) -> subprocess.Popen:
    '''
    Restarts the Streamlit app by killing the current process and starting a new one. Streamlit unfortunately doesn't have a built-in way to refresh the data.
    
    Args:
        streamlit_process: The current running process of Streamlit.
    '''
    print("Starting the Streamlit app...")
    if streamlit_process:
        streamlit_process.terminate()
        streamlit_process.wait()
    return run_streamlit_app()

def run_cpi_script_and_refresh(streamlit_process: subprocess.Popen):
    '''
    Runs the CPI data retrieval script and refreshes the Streamlit dashboard.
    
    Args:
        streamlit_process: The current running process of Streamlit.
    '''
    print(f"Triggering CPI data retrieval and dashboard refresh at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    run_cpi_script()  # update the data
    streamlit_process = restart_streamlit_app(streamlit_process)  # restart Streamlit to refresh data


def main():
    eastern_tz = pytz.timezone('US/Eastern')
    swiss_tz = pytz.timezone('Europe/Zurich')

    # BLS CPI data comes out at 08:30:00 EST. We take two seconds of buffer before running the API requests, in order to be sure of getting the new data. Convert then 08:30:02 EST to Switzerland time to properly running the scheduler.
    est_time = datetime.now(eastern_tz).replace(hour=8, minute=30, second=2, microsecond=0)
    swiss_time = est_time.astimezone(swiss_tz)

    # Run the CPI data retrieval script initially to have a Streamlit app with data to display even before the new data is available
    run_cpi_script()
    streamlit_process = run_streamlit_app()

    schedule_time_str = swiss_time.strftime("%H:%M:%S")
    print(f"Scheduling CPI data script to run every day at {schedule_time_str} (Switzerland time)")
    schedule.every().day.at(schedule_time_str).do(lambda: run_cpi_script_and_refresh(streamlit_process)) # mark the function to run every day at the specified time. A more advanced version would just run the script at the monthly data release date. But this is not needed for this project. It adds just a layer of complexity without having a different or better outcome.

    while True: # keep the script running to check the schedule
        current_time = datetime.now(eastern_tz)  # update current time to stay in sync with the schedule
        schedule.run_pending()
        print(f"Waiting... Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        time.sleep(1)

if __name__ == "__main__":
    main()