"""
Airtable Data Refresh Script
============================
Run this script to download fresh data from Airtable and save as CSV files.
The dashboard will then load from these CSVs for faster performance.

Usage: python refresh_airtable_data.py
"""

import requests
import pandas as pd
import os
from datetime import datetime

# Configuration
BASE_PATH = r"C:\Users\sunde\OneDrive\egmat files\2_Pulse Meeting Data\Pulse excels\streamlit-dashboard"
KEYS_FOLDER = os.path.join(BASE_PATH, "keys")
OUTPUT_FOLDER = os.path.join(BASE_PATH, "data")

def load_airtable_credentials():
    """Load Airtable credentials from keys folder"""
    try:
        with open(os.path.join(KEYS_FOLDER, 'airtable_api_key.txt'), 'r') as f:
            api_key = f.read().strip()
        with open(os.path.join(KEYS_FOLDER, 'airtable_base_id.txt'), 'r') as f:
            base_id = f.read().strip()
        with open(os.path.join(KEYS_FOLDER, 'Sales_Call_Tracker_table_name.txt'), 'r') as f:
            sales_table_id = f.read().strip()
        with open(os.path.join(KEYS_FOLDER, 'weekly_attendance_status_table_name.txt'), 'r') as f:
            attendance_table_id = f.read().strip()

        return {
            'api_key': api_key,
            'base_id': base_id,
            'sales_table_id': sales_table_id,
            'attendance_table_id': attendance_table_id
        }
    except FileNotFoundError as e:
        print(f"Error: Credential file not found: {e}")
        return None

def fetch_airtable_data(api_key, base_id, table_id, table_name):
    """Fetch all records from an Airtable table"""
    url = f"https://api.airtable.com/v0/{base_id}/{table_id}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    all_records = []
    offset = None
    page = 1

    print(f"  Fetching {table_name}...")

    while True:
        params = {}
        if offset:
            params['offset'] = offset

        response = requests.get(url, headers=headers, params=params)

        if response.status_code != 200:
            print(f"  Error: Airtable API returned {response.status_code} - {response.text}")
            return None

        data = response.json()
        records = data.get('records', [])
        all_records.extend(records)
        print(f"    Page {page}: Fetched {len(records)} records (Total: {len(all_records)})")

        offset = data.get('offset')
        page += 1
        if not offset:
            break

    # Convert to DataFrame
    if all_records:
        rows = []
        for record in all_records:
            row = record.get('fields', {})
            rows.append(row)
        return pd.DataFrame(rows)

    return pd.DataFrame()

def main():
    print("=" * 60)
    print("Airtable Data Refresh Script")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load credentials
    print("Loading Airtable credentials...")
    credentials = load_airtable_credentials()
    if credentials is None:
        print("Failed to load credentials. Exiting.")
        return False

    print("Credentials loaded successfully.")
    print()

    # Fetch Sales Call Tracker data
    print("1. Downloading Sales Call Tracker data...")
    sales_df = fetch_airtable_data(
        credentials['api_key'],
        credentials['base_id'],
        credentials['sales_table_id'],
        "Sales Call Tracker"
    )

    if sales_df is None or len(sales_df) == 0:
        print("  Failed to fetch Sales Call Tracker data!")
        return False

    # Save Sales Call Tracker CSV
    sales_csv_path = os.path.join(OUTPUT_FOLDER, 'Sales_Call_Tracker-Grid view.csv')
    sales_df.to_csv(sales_csv_path, index=False)
    print(f"  Saved: {sales_csv_path}")
    print(f"  Records: {len(sales_df)}")
    print()

    # Fetch Weekly Attendance Status data
    print("2. Downloading Weekly Attendance Status data...")
    attendance_df = fetch_airtable_data(
        credentials['api_key'],
        credentials['base_id'],
        credentials['attendance_table_id'],
        "Weekly Attendance Status"
    )

    if attendance_df is None or len(attendance_df) == 0:
        print("  Failed to fetch Weekly Attendance Status data!")
        return False

    # Save Weekly Attendance Status CSV
    attendance_csv_path = os.path.join(OUTPUT_FOLDER, 'weekly_attendance_status-Grid view.csv')
    attendance_df.to_csv(attendance_csv_path, index=False)
    print(f"  Saved: {attendance_csv_path}")
    print(f"  Records: {len(attendance_df)}")
    print()

    print("=" * 60)
    print("Data refresh completed successfully!")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()
    print("You can now run the dashboard with fresh data:")
    print("  python -m streamlit run new_dashboard.py")
    print()

    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
