import pandas as pd
import requests
import json
import csv
from datetime import datetime, timedelta

from tqdm import tqdm

def get_data(sensor_id, start_time, end_time):
    url = "https://aws.southcoastscience.com/topicMessages?topic=south-coast-science-test/afri-set/loc/%s/particulates&startTime=%sZ&endTime=%sZ" %(sensor_id, start_time.isoformat(), end_time.isoformat())
    headers = {'Authorization': 'api_key south-coast-science-test'}
    response = requests.get(url, headers=headers)
    data = response.json()
    # print(data)

    json_data = data["Items"]
    df = pd.DataFrame(pd.json_normalize(json_data))
    # print(df.head())
    return df
# print(headers)


def save_data(sensor_id, start_time, end_time):
    all_data = []
    total_items = 0
    pbar = tqdm(desc="Fetching data", unit="items")
    
    current_start_time = start_time

    try:
        while current_start_time < end_time:
            current_end_time = min(current_start_time + timedelta(days=4), end_time)
            offset = 0

            while True:
                url = f"https://aws.southcoastscience.com/topicMessages?topic=south-coast-science-test/afri-set/loc/{sensor_id}/particulates&startTime={current_start_time.isoformat()}Z&endTime={current_end_time.isoformat()}Z&offset={offset}&minMax=true&checkpoint=**:/5:00"
                headers = {'Authorization': 'api_key south-coast-science-test'}
                response = requests.get(url, headers=headers)
                data = response.json()
                
                json_data = data.get("Items", [])
                if not json_data:
                    break
                
                all_data.extend(json_data)
                offset += len(json_data)  # Update the offset to get the next set of data
                total_items += len(json_data)
                pbar.update(len(json_data))  # Update the progress bar

            current_start_time = current_end_time  # Move to the next chunk

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        pbar.close()
        df = pd.DataFrame(pd.json_normalize(all_data))
        return df

def main():
    sensor_id = "868"
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    end_time = datetime(2024, 10, 24, 23, 59, 59)
    df = save_data(sensor_id, start_time, end_time)
    with open("data.csv", "w") as f:
        df.to_csv(f)
    print(df)

if __name__ == "__main__":
    main()