import os
import time
import schedule
import subprocess
import streamlit as st
from datetime import datetime, timedelta
import pytz

# Function to execute the CPI data retrieval and processing script
def run_cpi_script():
    '''
    Run the CPI data retrieval and processing script.
    '''
    script_path = "CPI.py"
    subprocess.run(["python", script_path])

def run_streamlit_app():
    '''
    Function to run the Streamlit app.
    '''
    app_path = "Dashboard.py"
    subprocess.Popen(["streamlit", "run", app_path])

def main():
    switzerland_tz = pytz.timezone('Europe/Zurich')
    current_time = datetime.now(switzerland_tz)

    # Run the CPI data retrieval and processing script initially
    run_cpi_script()

    # Start the Streamlit app and keep the process reference
    streamlit_process = run_streamlit_app()

    # Check if today is August 14th
    if current_time.month == 8 and current_time.day == 14:
        # Schedule the CPI script to run at 13:00:05 Zurich time (13:00 + 5 seconds)
        schedule_time = (current_time.replace(hour=13, minute=0, second=0, microsecond=0) + timedelta(seconds=5)).time()
        schedule_time_str = schedule_time.strftime("%H:%M:%S")
        print(f"Scheduling CPI data script to run at {schedule_time_str}")

        def scheduled_task():
            run_cpi_script()
            # Restart the Streamlit app
            print("Restarting Streamlit app...")
            streamlit_process.terminate()  # Terminate the existing Streamlit process
            streamlit_process.wait()  # Wait for the process to terminate
            run_streamlit_app()  # Start the Streamlit app again

        schedule.every().day.at(schedule_time_str).do(scheduled_task)

    # Run the scheduling loop
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()
