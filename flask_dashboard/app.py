from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
import requests

app = Flask(__name__)
app.secret_key = 'sales_dashboard_secret_key'
app.config['APPLICATION_ROOT'] = '/sales/weeklymetrics'

# Support for running behind a proxy with URL prefix
from werkzeug.middleware.proxy_fix import ProxyFix
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Base paths
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_PATH, "data")
KEYS_PATH = os.path.join(BASE_PATH, "keys")
CONFIG_PATH = os.path.join(BASE_PATH, "config")
FILTERS_FILE = os.path.join(CONFIG_PATH, "global_filters.json")

# =============================================================================
# SLIDE TO FILTER TYPE MAPPING
# =============================================================================
# Metrics slides: Short date ranges (days/weeks) - for regular metrics, tables, percentages
# Trends slides: Longer date ranges (months) - for trend charts and analysis

METRICS_SLIDES = ['1', '2', '2.1', '2.2', '3', '7', '8']  # Overview, Attendance, Sales, Phone, Source
TRENDS_SLIDES = ['4', '5', '6', '6.1', '9']  # Bookings Trend, Attendance Trend, Conversion, Monthly
STATIC_SLIDES = ['10']  # Marketing - no date filter needed
DAILY_TRENDS_SLIDES = ['daily-1', 'daily-2']  # Daily Trends - independent filters

# =============================================================================
# GLOBAL FILTER MANAGEMENT
# =============================================================================

def load_global_filters():
    """Load global filters from JSON config file"""
    try:
        if os.path.exists(FILTERS_FILE):
            with open(FILTERS_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading filters: {e}")

    # Return defaults if file doesn't exist or error
    return {
        "metrics_filter": {
            "start_date": "2025-12-08",
            "end_date": "2025-12-14"
        },
        "trends_filter": {
            "start_date": "2025-02-10",
            "end_date": "2025-12-14"
        },
        "daily_trends_filter": {
            "calls_purchases": {
                "start_date": "2025-12-01",
                "end_date": "2025-12-14"
            },
            "attendance_bookings": {
                "start_date": "2025-12-01",
                "end_date": "2025-12-14"
            }
        },
        "last_updated": datetime.now().isoformat()
    }

def save_global_filters(filters):
    """Save global filters to JSON config file"""
    try:
        filters['last_updated'] = datetime.now().isoformat()
        with open(FILTERS_FILE, 'w') as f:
            json.dump(filters, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving filters: {e}")
        return False

def get_filter_for_slide(slide, filters):
    """Get the appropriate filter (metrics or trends) for a given slide"""
    if slide in STATIC_SLIDES:
        return None, None
    elif slide in TRENDS_SLIDES:
        return filters['trends_filter']['start_date'], filters['trends_filter']['end_date']
    else:
        # Default to metrics filter
        return filters['metrics_filter']['start_date'], filters['metrics_filter']['end_date']

def get_filter_type_for_slide(slide):
    """Get which filter type applies to a slide"""
    if slide in STATIC_SLIDES:
        return 'static'
    elif slide in TRENDS_SLIDES:
        return 'trends'
    else:
        return 'metrics'

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_airtable_credentials():
    """Load Airtable credentials from keys folder"""
    try:
        with open(os.path.join(KEYS_PATH, 'airtable_api_key.txt'), 'r') as f:
            api_key = f.read().strip()
        with open(os.path.join(KEYS_PATH, 'airtable_base_id.txt'), 'r') as f:
            base_id = f.read().strip()
        with open(os.path.join(KEYS_PATH, 'Sales_Call_Tracker_table_name.txt'), 'r') as f:
            sales_table_id = f.read().strip()
        with open(os.path.join(KEYS_PATH, 'weekly_attendance_status_table_name.txt'), 'r') as f:
            attendance_table_id = f.read().strip()
        return {
            'api_key': api_key,
            'base_id': base_id,
            'sales_table_id': sales_table_id,
            'attendance_table_id': attendance_table_id
        }
    except FileNotFoundError as e:
        return None

def fetch_airtable_data(api_key, base_id, table_id):
    """Fetch all records from an Airtable table"""
    url = f"https://api.airtable.com/v0/{base_id}/{table_id}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    all_records = []
    offset = None

    while True:
        params = {}
        if offset:
            params['offset'] = offset
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            return None
        data = response.json()
        records = data.get('records', [])
        all_records.extend(records)
        offset = data.get('offset')
        if not offset:
            break

    if all_records:
        rows = []
        for record in all_records:
            row = record.get('fields', {})
            row['record_id'] = record.get('id')
            rows.append(row)
        return pd.DataFrame(rows)
    return pd.DataFrame()

def refresh_data_from_airtable():
    """Fetch fresh data from Airtable and save as CSV files"""
    credentials = load_airtable_credentials()
    if credentials is None:
        return False, "Failed to load Airtable credentials"

    sales_df = fetch_airtable_data(
        credentials['api_key'],
        credentials['base_id'],
        credentials['sales_table_id']
    )
    if sales_df is None or len(sales_df) == 0:
        return False, "Failed to fetch Sales Call Tracker data"

    attendance_df = fetch_airtable_data(
        credentials['api_key'],
        credentials['base_id'],
        credentials['attendance_table_id']
    )
    if attendance_df is None or len(attendance_df) == 0:
        return False, "Failed to fetch Attendance data"

    sales_df.to_csv(os.path.join(DATA_PATH, 'Sales_Call_Tracker-Grid view.csv'), index=False)
    attendance_df.to_csv(os.path.join(DATA_PATH, 'weekly_attendance_status-Grid view.csv'), index=False)

    return True, f"Data refreshed! Sales: {len(sales_df)} records, Attendance: {len(attendance_df)} records"

def load_sales_data():
    """Load sales data from CSV"""
    df = pd.read_csv(os.path.join(DATA_PATH, 'Sales_Call_Tracker-Grid view.csv'))
    df['Call_Done_Date'] = pd.to_datetime(df.get('Call_Done_Date'), errors='coerce')
    if df['Call_Done_Date'].dt.tz is not None:
        df['Call_Done_Date'] = df['Call_Done_Date'].dt.tz_localize(None)
    df['Purchase_Intent'] = df.get('Purchase_Intent', pd.Series()).fillna('')
    df['Agent_Name'] = df.get('Agent_Name', pd.Series()).fillna('')
    df['Purchase_Status'] = df.get('Purchase_Status', pd.Series()).fillna('')
    df['Call_Type'] = df.get('Call_Type', pd.Series()).fillna('')
    df = df.dropna(subset=['Call_Done_Date'])
    return df

def load_attendance_data():
    """Load attendance data from CSV"""
    df = pd.read_csv(os.path.join(DATA_PATH, 'weekly_attendance_status-Grid view.csv'))
    df['EventDate'] = pd.to_datetime(df.get('Event_Date_'), errors='coerce')
    if df['EventDate'].dt.tz is not None:
        df['EventDate'] = df['EventDate'].dt.tz_localize(None)

    df['Call_Status'] = df.get('Call_Status', pd.Series()).fillna('')
    df['Whatsapp_Response'] = df.get('Whatsapp_Response', pd.Series()).fillna('')
    df['Source'] = df.get('Source', pd.Series()).fillna('')
    df['Sales_Agent_Name'] = df.get('Sales_Agent_Name', pd.Series()).fillna('')
    df['Region'] = df.get('Region', pd.Series()).fillna('')
    df['Prop_Type'] = df.get('Prop_Type', pd.Series()).fillna('')
    df['Email'] = df.get('Email', pd.Series()).fillna('')
    df['Contact_Number'] = df.get('Contact_Number', pd.Series()).fillna('')
    df['Probability_of_Response'] = df.get('Probability_of_Response', pd.Series())
    df['Profile_Cat_Prob_Resp'] = df.get('Profile_Cat_Prob_Resp', pd.Series()).fillna('')

    df = df.dropna(subset=['EventDate'])
    return df

# =============================================================================
# CALCULATION FUNCTIONS
# =============================================================================

def get_response_status(row):
    """Determine response status for a row"""
    call_status = str(row['Call_Status']).lower().strip()
    whatsapp = str(row['Whatsapp_Response']).lower().strip()
    if call_status == 'done' or whatsapp in ['in touch', 'confirmed attendance']:
        return 'Yes'
    elif whatsapp == 'no response':
        return 'No'
    return 'Blank'

def filter_data(df, start_date, end_date, date_col='EventDate', exclude_inbound=True, exclude_cant_happen=False):
    """Filter dataframe by date range and optionally exclude Inbound_Direct and Can't Happen records"""
    filtered = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)].copy()
    if exclude_inbound and 'Source' in filtered.columns:
        filtered = filtered[filtered['Source'].str.lower() != 'inbound_direct']
    # Exclude "Not Done + Africa" and "Can't Happen" - matching Streamlit's attendance_processed_filtered
    if exclude_cant_happen and 'Call_Status' in filtered.columns and 'Region' in filtered.columns:
        filtered = filtered[
            ~((filtered['Call_Status'].str.lower() == 'not done') & (filtered['Region'].str.lower() == 'africa')) &
            ~(filtered['Call_Status'].str.lower() == "can't happen")
        ]
    return filtered

# Slide 1: Overview
def calc_overview_metrics(sales_df, attendance_df, start_date, end_date):
    sales = filter_data(sales_df, start_date, end_date, 'Call_Done_Date', exclude_inbound=False)
    # Exclude Inbound_Direct for outbound attendance rate
    attendance = filter_data(attendance_df, start_date, end_date, 'EventDate', exclude_inbound=True)

    # Remove Can't Happen and Not Done + Africa
    attendance = attendance[
        ~((attendance['Call_Status'].str.lower() == 'not done') & (attendance['Region'].str.lower() == 'africa')) &
        ~(attendance['Call_Status'].str.lower() == "can't happen")
    ]

    calls_done = len(sales)
    high_intent = len(sales[sales['Purchase_Intent'].str.lower() == 'high'])
    high_call_rating = round((high_intent / calls_done * 100) if calls_done > 0 else 0)

    done_calls = len(attendance[attendance['Call_Status'].str.lower() == 'done'])
    outbound_attendance_rate = round((done_calls / len(attendance) * 100) if len(attendance) > 0 else 0)

    purchased = len(sales[sales['Purchase_Status'].str.lower() == 'purchased'])
    purchase_rate = round((purchased / calls_done * 100) if calls_done > 0 else 0)

    return {
        'calls_done': calls_done,
        'high_call_rating': high_call_rating,
        'outbound_attendance_rate': outbound_attendance_rate,
        'purchase_rate': purchase_rate,
        'purchases': purchased
    }

# Slide 2: Agent Attendance
def calc_agent_attendance(attendance_df, start_date, end_date):
    # Exclude Inbound_Direct
    filtered = filter_data(attendance_df, start_date, end_date, 'EventDate', exclude_inbound=True)
    filtered = filtered[
        ~((filtered['Call_Status'].str.lower() == 'not done') & (filtered['Region'].str.lower() == 'africa')) &
        ~(filtered['Call_Status'].str.lower() == "can't happen")
    ]

    agents = []
    for agent in filtered['Sales_Agent_Name'].unique():
        if not agent:
            continue
        agent_data = filtered[filtered['Sales_Agent_Name'] == agent]
        total = len(agent_data)
        done = len(agent_data[agent_data['Call_Status'].str.lower() == 'done'])
        rate = round((done / total * 100) if total > 0 else 0)

        good_data = agent_data[agent_data['Prop_Type'].str.lower().isin(['good prop', 'good prospect'])]
        good_total = len(good_data)
        good_done = len(good_data[good_data['Call_Status'].str.lower() == 'done'])
        good_rate = round((good_done / good_total * 100) if good_total > 0 else 0)

        agents.append({
            'agent': agent,
            'bookings': total,
            'attendance_rate': rate,
            'good_prospect_rate': good_rate
        })
    return sorted(agents, key=lambda x: x['agent'])

# Slide 2.1: Attendance Breakdown
def calc_attendance_breakdown(attendance_df, start_date, end_date):
    """Calculate attendance breakdown metrics for slide 2.1"""
    # Exclude Inbound_Direct
    filtered = filter_data(attendance_df, start_date, end_date, 'EventDate', exclude_inbound=True)
    if len(filtered) == 0:
        return None

    # Row 1 metrics
    calls_booked = len(filtered)
    calls_cant_happen = len(filtered[filtered['Call_Status'].str.lower() == "can't happen"])
    calls_done = len(filtered[filtered['Call_Status'].str.lower() == 'done'])
    attendance_rate = round((calls_done / calls_booked * 100) if calls_booked > 0 else 0, 1)

    # No Response Rate - those with Whatsapp_Response = 'No Response' out of all bookings
    no_response_count = len(filtered[filtered['Whatsapp_Response'].str.lower() == 'no response'])
    no_response_rate = round((no_response_count / calls_booked * 100) if calls_booked > 0 else 0, 1)

    # Row 2 metrics - Breakdown of Calls Not Done by Whatsapp_Response
    not_done = filtered[filtered['Call_Status'].str.lower() == 'not done']
    whatsapp_no_response = len(not_done[not_done['Whatsapp_Response'].str.lower() == 'no response'])
    whatsapp_in_touch = len(not_done[not_done['Whatsapp_Response'].str.lower() == 'in touch'])
    whatsapp_confirmed = len(not_done[not_done['Whatsapp_Response'].str.lower() == 'confirmed attendance'])

    return {
        'calls_booked': calls_booked,
        'calls_cant_happen': calls_cant_happen,
        'calls_done': calls_done,
        'attendance_rate': attendance_rate,
        'no_response_rate': no_response_rate,
        'no_response_count': no_response_count,
        'whatsapp_no_response': whatsapp_no_response,
        'whatsapp_in_touch': whatsapp_in_touch,
        'whatsapp_confirmed': whatsapp_confirmed
    }

def calc_in_touch_not_done(attendance_df, start_date, end_date):
    """Get users where Call_Status is Not Done but Whatsapp_Response is In Touch"""
    # Exclude Inbound_Direct
    filtered = filter_data(attendance_df, start_date, end_date, 'EventDate', exclude_inbound=True)

    # Filter for Not Done + In Touch
    in_touch_not_done = filtered[
        (filtered['Call_Status'].str.lower() == 'not done') &
        (filtered['Whatsapp_Response'].str.lower() == 'in touch')
    ].copy()

    records = []
    for _, row in in_touch_not_done.iterrows():
        # Convert to string and handle NaN values
        not_done_reason = row.get('Not_Done_Reason_Category', '')
        agent = row.get('Sales_Agent_Name', '')
        interaction = row.get('Interaction_Explanation', '')

        records.append({
            'email': row.get('Email', ''),
            'agent': str(agent) if pd.notna(agent) else '',
            'not_done_reason': str(not_done_reason) if pd.notna(not_done_reason) else '',
            'interaction_explanation': str(interaction) if pd.notna(interaction) else ''
        })

    # Sort by Not_Done_Reason_Category
    return sorted(records, key=lambda x: (x['not_done_reason'], x['agent']))

# Slide 2.2: Response Rate Tracker
def calc_response_metrics(attendance_df, start_date, end_date):
    filtered = filter_data(attendance_df, start_date, end_date)
    if len(filtered) == 0:
        return None

    filtered['Response_Status'] = filtered.apply(get_response_status, axis=1)
    calls_booked = len(filtered)
    yes = len(filtered[filtered['Response_Status'] == 'Yes'])
    no = len(filtered[filtered['Response_Status'] == 'No'])
    blank = len(filtered[filtered['Response_Status'] == 'Blank'])
    response_rate = round((yes / calls_booked * 100) if calls_booked > 0 else 0, 1)
    calls_done = len(filtered[filtered['Call_Status'].str.lower() == 'done'])
    attendance_rate = round((calls_done / calls_booked * 100) if calls_booked > 0 else 0, 1)

    return {
        'calls_booked': calls_booked,
        'responses_yes': yes,
        'responses_no': no,
        'responses_blank': blank,
        'response_rate': response_rate,
        'calls_done': calls_done,
        'attendance_rate': attendance_rate
    }

def calc_daily_response(attendance_df, start_date, end_date):
    filtered = filter_data(attendance_df, start_date, end_date)
    if len(filtered) == 0:
        return []

    filtered['Response_Status'] = filtered.apply(get_response_status, axis=1)
    filtered['DateOnly'] = filtered['EventDate'].dt.date

    daily = []
    for date in sorted(filtered['DateOnly'].unique()):
        day = filtered[filtered['DateOnly'] == date]
        booked = len(day)
        yes = len(day[day['Response_Status'] == 'Yes'])
        done = len(day[day['Call_Status'].str.lower() == 'done'])
        daily.append({
            'date': date.strftime('%Y-%m-%d'),
            'calls_booked': booked,
            'responded': yes,
            'response_rate': round((yes / booked * 100) if booked > 0 else 0, 1),
            'calls_done': done,
            'attendance_rate': round((done / booked * 100) if booked > 0 else 0, 1)
        })
    return daily

def calc_agent_response(attendance_df, start_date, end_date):
    filtered = filter_data(attendance_df, start_date, end_date)
    if len(filtered) == 0:
        return []

    filtered['Response_Status'] = filtered.apply(get_response_status, axis=1)

    agents = []
    for agent in sorted(filtered['Sales_Agent_Name'].unique()):
        if not agent or not str(agent).strip():
            continue
        agent_data = filtered[filtered['Sales_Agent_Name'] == agent]
        booked = len(agent_data)
        yes = len(agent_data[agent_data['Response_Status'] == 'Yes'])
        no = len(agent_data[agent_data['Response_Status'] == 'No'])
        blank = len(agent_data[agent_data['Response_Status'] == 'Blank'])
        done = len(agent_data[agent_data['Call_Status'].str.lower() == 'done'])

        agents.append({
            'agent': agent,
            'calls_booked': booked,
            'responses_yes': yes,
            'responses_no': no,
            'responses_blank': blank,
            'response_rate': round((yes / booked * 100) if booked > 0 else 0, 1),
            'calls_done': done,
            'attendance_rate': round((done / booked * 100) if booked > 0 else 0, 1)
        })
    return agents

def calc_model_accuracy(attendance_df, start_date, end_date):
    filtered = filter_data(attendance_df, start_date, end_date)
    if len(filtered) == 0:
        return None

    total = len(filtered)
    prob_exists = filtered['Probability_of_Response'].notna().sum()
    prob_blank = total - prob_exists

    filtered['Response_Status'] = filtered.apply(get_response_status, axis=1)
    with_prob = filtered[filtered['Probability_of_Response'].notna()].copy()

    breakdown = []
    for profile in with_prob['Profile_Cat_Prob_Resp'].unique():
        if not profile or str(profile).strip() == '':
            continue
        pdata = with_prob[with_prob['Profile_Cat_Prob_Resp'] == profile]
        count = len(pdata)

        prob_raw = pdata['Probability_of_Response'].mode()
        if len(prob_raw) > 0:
            try:
                model_prob = float(str(prob_raw.iloc[0]).replace('%', ''))
            except:
                model_prob = 0
        else:
            model_prob = 0

        yes = len(pdata[pdata['Response_Status'] == 'Yes'])
        actual_rate = round((yes / count * 100) if count > 0 else 0, 1)

        breakdown.append({
            'profile': profile,
            'count': count,
            'model_prob': model_prob,
            'actual_rate': actual_rate,
            'difference': round(actual_rate - model_prob, 1)
        })

    return {
        'total': total,
        'prob_exists': prob_exists,
        'prob_blank': prob_blank,
        'breakdown': breakdown
    }

def calc_blank_response(attendance_df, start_date, end_date):
    filtered = filter_data(attendance_df, start_date, end_date)
    blank = filtered[
        (filtered['Call_Status'].str.lower() == 'not done') &
        ((filtered['Whatsapp_Response'] == '') | (filtered['Whatsapp_Response'].isna()))
    ].copy()

    records = []
    for _, row in blank.iterrows():
        records.append({
            'email': row['Email'],
            'agent': row['Sales_Agent_Name'],
            'date': row['EventDate'].strftime('%Y-%m-%d') if pd.notna(row['EventDate']) else '',
            'phone': row['Contact_Number']
        })
    return sorted(records, key=lambda x: x['date'], reverse=True)

# Slide 3: Agent Sales
def calc_agent_sales(sales_df, start_date, end_date):
    filtered = filter_data(sales_df, start_date, end_date, 'Call_Done_Date', exclude_inbound=False)

    agents = []
    for agent in filtered['Agent_Name'].unique():
        if not agent:
            continue
        agent_data = filtered[filtered['Agent_Name'] == agent]
        calls = len(agent_data)
        purchases = len(agent_data[agent_data['Purchase_Status'].str.lower() == 'purchased'])
        high = len(agent_data[agent_data['Purchase_Intent'].str.lower() == 'high'])

        agents.append({
            'agent': agent,
            'calls_done': calls,
            'purchases': purchases,
            'high_rating': round((high / calls * 100) if calls > 0 else 0),
            'purchase_rate': round((purchases / calls * 100) if calls > 0 else 0)
        })
    return sorted(agents, key=lambda x: x['agent'])

# Helper function to calculate week (Monday to Sunday) - matching Streamlit exactly
def calculate_week_monday_sunday(date):
    """Calculate week label as 'Week of YYYY-MM-DD' where date is the Monday of that week"""
    if pd.isna(date):
        return ''
    # Get the Monday of the week containing this date
    monday = date - pd.Timedelta(days=date.weekday())
    return f"Week of {monday.strftime('%Y-%m-%d')}"

# Slide 4: Bookings Trend (excluding Inbound_Direct)
def calc_bookings_trend(attendance_df, start_date, end_date):
    # Exclude Inbound_Direct and Can't Happen
    filtered = filter_data(attendance_df, start_date, end_date, 'EventDate', exclude_inbound=True, exclude_cant_happen=True)
    if len(filtered) == 0:
        return []

    # Get Cohort column (mapped from Prop_Type)
    filtered['Cohort'] = filtered['Prop_Type'].fillna('').replace({
        'Good Prop': 'Good Prospect',
        'Good prop': 'Good Prospect',
        'Raw Prop': 'Raw Prospect',
        'Raw prop': 'Raw Prospect'
    })

    # Create Week_Of_Call column using same format as Streamlit: "Week of YYYY-MM-DD"
    filtered['Week_Of_Call'] = filtered['EventDate'].apply(calculate_week_monday_sunday)

    trend_data = []
    for week in filtered['Week_Of_Call'].unique():
        if not week or pd.isna(week) or week == '':
            continue
        week_data = filtered[filtered['Week_Of_Call'] == week]
        total = len(week_data)
        raw = len(week_data[week_data['Cohort'].str.lower() == 'raw prospect'])
        good = len(week_data[week_data['Cohort'].str.lower() == 'good prospect'])
        earliest = week_data['EventDate'].min()

        trend_data.append({
            'week': str(week),
            'total_bookings': total,
            'raw_prospect': raw,
            'good_prospect': good,
            'sort_date': earliest
        })

    return sorted(trend_data, key=lambda x: x['sort_date'])

# Slide 4: Inbound_Direct Bookings Trend (only Inbound_Direct)
def calc_inbound_bookings_trend(attendance_df, start_date, end_date):
    """Calculate bookings trend for Inbound_Direct only"""
    # Filter by date range first
    filtered = attendance_df[(attendance_df['EventDate'] >= start_date) & (attendance_df['EventDate'] <= end_date)].copy()
    # Keep only Inbound_Direct
    filtered = filtered[filtered['Source'].str.lower() == 'inbound_direct']
    # Exclude Can't Happen
    filtered = filtered[
        ~((filtered['Call_Status'].str.lower() == 'not done') & (filtered['Region'].str.lower() == 'africa')) &
        ~(filtered['Call_Status'].str.lower() == "can't happen")
    ]

    if len(filtered) == 0:
        return []

    # Create Week_Of_Call column
    filtered['Week_Of_Call'] = filtered['EventDate'].apply(calculate_week_monday_sunday)

    trend_data = []
    for week in filtered['Week_Of_Call'].unique():
        if not week or pd.isna(week) or week == '':
            continue
        week_data = filtered[filtered['Week_Of_Call'] == week]
        total = len(week_data)
        earliest = week_data['EventDate'].min()

        trend_data.append({
            'week': str(week),
            'total_bookings': total,
            'sort_date': earliest
        })

    return sorted(trend_data, key=lambda x: x['sort_date'])

# Slide 5: Attendance Trend
def calc_attendance_trend(attendance_df, start_date, end_date):
    # Exclude Inbound_Direct and Can't Happen
    filtered = filter_data(attendance_df, start_date, end_date, 'EventDate', exclude_inbound=True, exclude_cant_happen=True)
    if len(filtered) == 0:
        return []

    filtered['Cohort'] = filtered['Prop_Type'].fillna('').replace({
        'Good Prop': 'Good Prospect',
        'Good prop': 'Good Prospect',
        'Raw Prop': 'Raw Prospect',
        'Raw prop': 'Raw Prospect'
    })

    # Create Week_Of_Call column using same format as Streamlit: "Week of YYYY-MM-DD"
    filtered['Week_Of_Call'] = filtered['EventDate'].apply(calculate_week_monday_sunday)

    trend_data = []
    for week in filtered['Week_Of_Call'].unique():
        if not week or pd.isna(week) or week == '':
            continue
        week_data = filtered[filtered['Week_Of_Call'] == week]

        total = len(week_data)
        done = len(week_data[week_data['Call_Status'].str.lower() == 'done'])
        overall_rate = round((done / total * 100) if total > 0 else 0)

        raw_data = week_data[week_data['Cohort'].str.lower() == 'raw prospect']
        raw_total = len(raw_data)
        raw_done = len(raw_data[raw_data['Call_Status'].str.lower() == 'done'])
        raw_rate = round((raw_done / raw_total * 100) if raw_total > 0 else 0)

        good_data = week_data[week_data['Cohort'].str.lower() == 'good prospect']
        good_total = len(good_data)
        good_done = len(good_data[good_data['Call_Status'].str.lower() == 'done'])
        good_rate = round((good_done / good_total * 100) if good_total > 0 else 0)

        earliest = week_data['EventDate'].min()

        trend_data.append({
            'week': str(week),
            'overall_rate': overall_rate,
            'raw_rate': raw_rate,
            'good_rate': good_rate,
            'sort_date': earliest
        })

    return sorted(trend_data, key=lambda x: x['sort_date'])

# Slide 5: Response Rate Trend
def calc_response_rate_trend(attendance_df, start_date, end_date):
    """Calculate response rate trend (Call Done OR In Touch OR Confirmed Attendance)"""
    # Exclude Inbound_Direct and Can't Happen
    filtered = filter_data(attendance_df, start_date, end_date, 'EventDate', exclude_inbound=True, exclude_cant_happen=True)
    if len(filtered) == 0:
        return []

    # Create Week_Of_Call column
    filtered['Week_Of_Call'] = filtered['EventDate'].apply(calculate_week_monday_sunday)

    trend_data = []
    for week in filtered['Week_Of_Call'].unique():
        if not week or pd.isna(week) or week == '':
            continue
        week_data = filtered[filtered['Week_Of_Call'] == week]

        total = len(week_data)

        # Response = Call Done OR In Touch OR Confirmed Attendance
        responded = len(week_data[
            (week_data['Call_Status'].str.lower() == 'done') |
            (week_data['Whatsapp_Response'].str.lower() == 'in touch') |
            (week_data['Whatsapp_Response'].str.lower() == 'confirmed attendance')
        ])
        response_rate = round((responded / total * 100) if total > 0 else 0)

        earliest = week_data['EventDate'].min()

        trend_data.append({
            'week': str(week),
            'total': total,
            'responded': responded,
            'response_rate': response_rate,
            'sort_date': earliest
        })

    return sorted(trend_data, key=lambda x: x['sort_date'])

# Slide 5: Overall Bookings Trend (Outbound + Inbound combined)
def calc_overall_bookings_trend(attendance_df, start_date, end_date):
    """Calculate overall bookings trend combining outbound + inbound, with calls done and attendance rate"""
    # Filter by date range - include ALL sources (both outbound and inbound)
    filtered = attendance_df[(attendance_df['EventDate'] >= start_date) & (attendance_df['EventDate'] <= end_date)].copy()

    # Exclude Can't Happen and Not Done + Africa (same exclusions as other charts)
    filtered = filtered[
        ~((filtered['Call_Status'].str.lower() == 'not done') & (filtered['Region'].str.lower() == 'africa')) &
        ~(filtered['Call_Status'].str.lower() == "can't happen")
    ]

    if len(filtered) == 0:
        return []

    # Create Week_Of_Call column
    filtered['Week_Of_Call'] = filtered['EventDate'].apply(calculate_week_monday_sunday)

    trend_data = []
    for week in filtered['Week_Of_Call'].unique():
        if not week or pd.isna(week) or week == '':
            continue
        week_data = filtered[filtered['Week_Of_Call'] == week]

        total_bookings = len(week_data)
        calls_done = len(week_data[week_data['Call_Status'].str.lower() == 'done'])
        attendance_rate = round((calls_done / total_bookings * 100) if total_bookings > 0 else 0)

        earliest = week_data['EventDate'].min()

        trend_data.append({
            'week': str(week),
            'total_bookings': total_bookings,
            'calls_done': calls_done,
            'attendance_rate': attendance_rate,
            'sort_date': earliest
        })

    return sorted(trend_data, key=lambda x: x['sort_date'])

# Slide 6: Conversion Trend
def calc_conversion_trend(sales_df, start_date, end_date):
    filtered = filter_data(sales_df, start_date, end_date, 'Call_Done_Date', exclude_inbound=False)
    if len(filtered) == 0:
        return []

    # Compute week's Monday from Call_Done_Date (Monday=0, so subtract weekday days)
    # Normalize to midnight to ensure proper grouping
    filtered = filtered.copy()
    filtered['Week_Monday'] = filtered['Call_Done_Date'].apply(
        lambda x: (x - timedelta(days=x.weekday())).normalize() if pd.notna(x) else None
    )

    # Group by week
    grouped = filtered.groupby('Week_Monday')

    trend_data = []
    for monday, week_data in grouped:
        if pd.isna(monday):
            continue

        calls = len(week_data)
        purchases = len(week_data[week_data['Purchase_Status'].str.lower() == 'purchased'])
        rate = round((purchases / calls * 100) if calls > 0 else 0)

        # Format label as "Mon DD" (e.g., "Dec 29")
        week_label = monday.strftime('%b %d')

        trend_data.append({
            'week': week_label,
            'calls_done': calls,
            'purchases': purchases,
            'purchase_rate': rate,
            'sort_date': monday
        })

    return sorted(trend_data, key=lambda x: x['sort_date'])

# Slide 6.1: Conversion Trend by Intent
def calc_conversion_trend_by_intent(sales_df, start_date, end_date):
    filtered = filter_data(sales_df, start_date, end_date, 'Call_Done_Date', exclude_inbound=False)
    if len(filtered) == 0:
        return {'high': [], 'medium': [], 'low': []}

    # Compute week's Monday from Call_Done_Date (Monday=0, so subtract weekday days)
    # Normalize to midnight to ensure proper grouping
    filtered = filtered.copy()
    filtered['Week_Monday'] = filtered['Call_Done_Date'].apply(
        lambda x: (x - timedelta(days=x.weekday())).normalize() if pd.notna(x) else None
    )

    # Get unique weeks sorted
    unique_weeks = sorted(filtered['Week_Monday'].dropna().unique())

    result = {'high': [], 'medium': [], 'low': []}

    for intent in ['high', 'medium', 'low']:
        intent_data = filtered[filtered['Purchase_Intent'].str.lower() == intent]
        trend_data = []

        for monday in unique_weeks:
            week_data = intent_data[intent_data['Week_Monday'] == monday]

            calls = len(week_data)
            purchases = len(week_data[week_data['Purchase_Status'].str.lower() == 'purchased'])
            rate = round((purchases / calls * 100) if calls > 0 else 0)

            # Format label as "Mon DD" (e.g., "Dec 29")
            week_label = monday.strftime('%b %d')

            trend_data.append({
                'week': week_label,
                'calls_done': calls,
                'purchases': purchases,
                'purchase_rate': rate,
                'sort_date': monday
            })

        result[intent] = trend_data  # Already sorted by unique_weeks order

    return result

# Slide 6.1: Calls Done and Low Intent Percent Trend
def calc_calls_low_intent_trend(sales_df, start_date, end_date):
    """Calculate weekly calls done and low intent percent"""
    filtered = filter_data(sales_df, start_date, end_date, 'Call_Done_Date', exclude_inbound=False)
    if len(filtered) == 0:
        return []

    # Compute week's Monday from Call_Done_Date (Monday=0, so subtract weekday days)
    # Normalize to midnight to ensure proper grouping
    filtered = filtered.copy()
    filtered['Week_Monday'] = filtered['Call_Done_Date'].apply(
        lambda x: (x - timedelta(days=x.weekday())).normalize() if pd.notna(x) else None
    )

    # Group by week
    grouped = filtered.groupby('Week_Monday')

    trend_data = []
    for monday, week_data in grouped:
        if pd.isna(monday):
            continue

        total_calls = len(week_data)
        low_intent_calls = len(week_data[week_data['Purchase_Intent'].str.lower() == 'low'])
        low_intent_percent = round((low_intent_calls / total_calls * 100) if total_calls > 0 else 0, 1)

        # Format label as "Mon DD" (e.g., "Dec 29")
        week_label = monday.strftime('%b %d')

        trend_data.append({
            'week': week_label,
            'calls_done': total_calls,
            'low_intent_count': low_intent_calls,
            'low_intent_percent': low_intent_percent,
            'sort_date': monday
        })

    return sorted(trend_data, key=lambda x: x['sort_date'])

# Slide 7: Phone Call Analysis
def calc_phone_call_metrics(sales_df, start_date, end_date):
    filtered = filter_data(sales_df, start_date, end_date, 'Call_Done_Date', exclude_inbound=False)
    # Filter for phone calls only
    filtered = filtered[filtered['Call_Type'].str.lower() == 'phone call']

    if len(filtered) == 0:
        return []

    agents = []
    for agent in filtered['Agent_Name'].unique():
        if not agent:
            continue
        agent_data = filtered[filtered['Agent_Name'] == agent]
        calls = len(agent_data)
        purchases = len(agent_data[agent_data['Purchase_Status'].str.lower() == 'purchased'])
        high = len(agent_data[agent_data['Purchase_Intent'].str.lower() == 'high'])

        agents.append({
            'agent': agent,
            'phone_calls_done': calls,
            'phone_call_purchases': purchases,
            'high_rating': round((high / calls * 100) if calls > 0 else 0),
            'purchase_rate': round((purchases / calls * 100) if calls > 0 else 0)
        })

    return sorted(agents, key=lambda x: x['agent'])

# Slide 8: Source Breakdown
def calc_source_breakdown(attendance_df, start_date, end_date):
    filtered = filter_data(attendance_df, start_date, end_date, 'EventDate', exclude_inbound=False)
    if len(filtered) == 0:
        return []

    sources = []
    for source in filtered['Source'].unique():
        if not source or pd.isna(source):
            continue
        source_data = filtered[filtered['Source'] == source]
        bookings = len(source_data)
        done = len(source_data[source_data['Call_Status'].str.lower() == 'done'])
        rate = round((done / bookings * 100) if bookings > 0 else 0)

        sources.append({
            'source': source,
            'bookings': bookings,
            'calls_done': done,
            'attendance_rate': rate
        })

    return sorted(sources, key=lambda x: x['bookings'], reverse=True)

def calc_source_agent_breakdown(attendance_df, start_date, end_date):
    filtered = filter_data(attendance_df, start_date, end_date, 'EventDate', exclude_inbound=False)
    if len(filtered) == 0:
        return []

    results = []
    for source in filtered['Source'].unique():
        if not source or pd.isna(source):
            continue
        source_data = filtered[filtered['Source'] == source]

        for agent in source_data['Sales_Agent_Name'].unique():
            if not agent:
                continue
            agent_data = source_data[source_data['Sales_Agent_Name'] == agent]
            bookings = len(agent_data)
            done = len(agent_data[agent_data['Call_Status'].str.lower() == 'done'])
            rate = round((done / bookings * 100) if bookings > 0 else 0)

            results.append({
                'source': source,
                'agent': agent,
                'bookings': bookings,
                'calls_done': done,
                'attendance_rate': rate
            })

    return sorted(results, key=lambda x: (x['source'], x['agent']))

def calc_leakage_metrics(attendance_df, start_date, end_date):
    # Use unfiltered data for leakage
    filtered = attendance_df[(attendance_df['EventDate'] >= start_date) & (attendance_df['EventDate'] <= end_date)].copy()

    total = len(filtered)
    cant_happen = len(filtered[
        ((filtered['Call_Status'].str.lower() == 'not done') & (filtered['Region'].str.lower() == 'africa')) |
        (filtered['Call_Status'].str.lower() == "can't happen")
    ])
    net = total - cant_happen
    leakage_rate = round((cant_happen / total * 100) if total > 0 else 0)

    return {
        'total_calls_booked': total,
        'cant_happen_calls': cant_happen,
        'net_bookings': net,
        'leakage_rate': leakage_rate
    }

def calc_booking_source_breakdown(sales_df, start_date, end_date):
    """Calculate booking source breakdown metrics from sales data"""
    filtered = filter_data(sales_df, start_date, end_date, 'Call_Done_Date', exclude_inbound=False)
    if len(filtered) == 0:
        return []

    results = []
    for booking_source in filtered['Booking_Source'].unique() if 'Booking_Source' in filtered.columns else []:
        if not booking_source or pd.isna(booking_source):
            continue
        source_data = filtered[filtered['Booking_Source'] == booking_source]
        calls = len(source_data)
        purchased = len(source_data[source_data['Purchase_Status'].str.lower() == 'purchased'])
        rate = round((purchased / calls * 100) if calls > 0 else 0)

        results.append({
            'booking_source': booking_source,
            'calls_done': calls,
            'purchased': purchased,
            'purchase_rate': rate
        })

    return sorted(results, key=lambda x: x['calls_done'], reverse=True)

# Slide 9: Monthly Sales
def calc_monthly_sales(sales_df, start_date, end_date):
    filtered = filter_data(sales_df, start_date, end_date, 'Call_Done_Date', exclude_inbound=False)
    if len(filtered) == 0:
        return []

    filtered['Month'] = filtered['Call_Done_Date'].dt.strftime('%Y-%m')

    monthly = []
    for month in sorted(filtered['Month'].unique()):
        month_data = filtered[filtered['Month'] == month]
        calls = len(month_data)
        purchases = len(month_data[month_data['Purchase_Status'].str.lower() == 'purchased'])
        rate = round((purchases / calls * 100) if calls > 0 else 0)

        monthly.append({
            'month': month,
            'calls_done': calls,
            'purchases': purchases,
            'purchase_rate': rate
        })

    return monthly

def calc_effective_attendance(sales_df, attendance_df, start_date, end_date):
    sales_filtered = filter_data(sales_df, start_date, end_date, 'Call_Done_Date', exclude_inbound=False)
    att_filtered = filter_data(attendance_df, start_date, end_date, 'EventDate', exclude_inbound=False)

    if len(sales_filtered) == 0 or len(att_filtered) == 0:
        return []

    sales_filtered['Month'] = sales_filtered['Call_Done_Date'].dt.strftime('%Y-%m')
    att_filtered['Month'] = att_filtered['EventDate'].dt.strftime('%Y-%m')

    all_months = set(sales_filtered['Month'].unique()) | set(att_filtered['Month'].unique())

    results = []
    for month in sorted(all_months):
        booked = len(att_filtered[att_filtered['Month'] == month])
        done = len(sales_filtered[sales_filtered['Month'] == month])
        rate = round((done / booked * 100) if booked > 0 else 0)

        results.append({
            'month': month,
            'calls_booked': booked,
            'calls_done': done,
            'effective_attendance_rate': rate
        })

    return results

# =============================================================================
# DAILY TRENDS CALCULATION FUNCTIONS
# =============================================================================

def calc_daily_calls_trend(sales_df, start_date, end_date):
    """Calculate daily calls done trend"""
    filtered = filter_data(sales_df, start_date, end_date, 'Call_Done_Date', exclude_inbound=False)
    if len(filtered) == 0:
        return []

    filtered['DateOnly'] = filtered['Call_Done_Date'].dt.date

    daily = []
    for date in sorted(filtered['DateOnly'].unique()):
        day_data = filtered[filtered['DateOnly'] == date]
        calls = len(day_data)
        daily.append({
            'date': date.strftime('%Y-%m-%d'),
            'day_name': date.strftime('%A'),
            'calls_done': calls
        })
    return daily

def calc_daily_purchases_trend(sales_df, start_date, end_date):
    """Calculate daily purchases and purchase rate trend"""
    filtered = filter_data(sales_df, start_date, end_date, 'Call_Done_Date', exclude_inbound=False)
    if len(filtered) == 0:
        return []

    filtered['DateOnly'] = filtered['Call_Done_Date'].dt.date

    daily = []
    for date in sorted(filtered['DateOnly'].unique()):
        day_data = filtered[filtered['DateOnly'] == date]
        calls = len(day_data)
        purchases = len(day_data[day_data['Purchase_Status'].str.lower() == 'purchased'])
        rate = round((purchases / calls * 100) if calls > 0 else 0, 1)
        daily.append({
            'date': date.strftime('%Y-%m-%d'),
            'day_name': date.strftime('%A'),
            'calls_done': calls,
            'purchases': purchases,
            'purchase_rate': rate
        })
    return daily

def calc_daily_attendance_trend(attendance_df, start_date, end_date):
    """Calculate daily attendance trend"""
    # Exclude Inbound_Direct and Can't Happen
    filtered = filter_data(attendance_df, start_date, end_date, 'EventDate', exclude_inbound=True, exclude_cant_happen=True)
    if len(filtered) == 0:
        return []

    filtered['DateOnly'] = filtered['EventDate'].dt.date

    daily = []
    for date in sorted(filtered['DateOnly'].unique()):
        day_data = filtered[filtered['DateOnly'] == date]
        total = len(day_data)
        done = len(day_data[day_data['Call_Status'].str.lower() == 'done'])
        rate = round((done / total * 100) if total > 0 else 0, 1)
        daily.append({
            'date': date.strftime('%Y-%m-%d'),
            'day_name': date.strftime('%A'),
            'bookings': total,
            'calls_done': done,
            'attendance_rate': rate
        })
    return daily

def calc_daily_bookings_trend(attendance_df, start_date, end_date):
    """Calculate daily bookings trend (new bookings per day)"""
    # Exclude Inbound_Direct and Can't Happen
    filtered = filter_data(attendance_df, start_date, end_date, 'EventDate', exclude_inbound=True, exclude_cant_happen=True)
    if len(filtered) == 0:
        return []

    # Get Cohort column (mapped from Prop_Type)
    filtered['Cohort'] = filtered['Prop_Type'].fillna('').replace({
        'Good Prop': 'Good Prospect',
        'Good prop': 'Good Prospect',
        'Raw Prop': 'Raw Prospect',
        'Raw prop': 'Raw Prospect'
    })

    filtered['DateOnly'] = filtered['EventDate'].dt.date

    daily = []
    for date in sorted(filtered['DateOnly'].unique()):
        day_data = filtered[filtered['DateOnly'] == date]
        total = len(day_data)
        raw = len(day_data[day_data['Cohort'].str.lower() == 'raw prospect'])
        good = len(day_data[day_data['Cohort'].str.lower() == 'good prospect'])
        daily.append({
            'date': date.strftime('%Y-%m-%d'),
            'day_name': date.strftime('%A'),
            'total_bookings': total,
            'raw_prospect': raw,
            'good_prospect': good
        })
    return daily

# =============================================================================
# DASHBOARD SNAPSHOT EXPORT
# =============================================================================

def export_dashboard_snapshot():
    """
    Export complete dashboard data snapshot to JSON file.
    Collects data from all slides using current filter settings.
    """
    # Load current filters
    global_filters = load_global_filters()

    # Load data
    sales_df = load_sales_data()
    attendance_df = load_attendance_data()

    # Calculate date ranges
    metrics_start = pd.to_datetime(global_filters['metrics_filter']['start_date'])
    metrics_end = pd.to_datetime(global_filters['metrics_filter']['end_date']) + timedelta(days=1) - timedelta(seconds=1)

    trends_start = pd.to_datetime(global_filters['trends_filter']['start_date'])
    trends_end = pd.to_datetime(global_filters['trends_filter']['end_date']) + timedelta(days=1) - timedelta(seconds=1)

    # Build snapshot
    snapshot = {
        'exported_at': datetime.now().isoformat(),
        'filters': {
            'metrics': {
                'start_date': global_filters['metrics_filter']['start_date'],
                'end_date': global_filters['metrics_filter']['end_date'],
                'display_range': f"{metrics_start.strftime('%b %d, %Y')} - {metrics_end.strftime('%b %d, %Y')}"
            },
            'trends': {
                'start_date': global_filters['trends_filter']['start_date'],
                'end_date': global_filters['trends_filter']['end_date'],
                'display_range': f"{trends_start.strftime('%b %d, %Y')} - {trends_end.strftime('%b %d, %Y')}"
            }
        },
        'slides': {}
    }

    # Slide 1: Overview (metrics filter)
    snapshot['slides']['overview'] = calc_overview_metrics(sales_df, attendance_df, metrics_start, metrics_end)

    # Slide 2: Agent Attendance (metrics filter)
    agent_attendance = calc_agent_attendance(attendance_df, metrics_start, metrics_end)
    # Also calculate team-wide metrics
    filtered = attendance_df[(attendance_df['EventDate'] >= metrics_start) & (attendance_df['EventDate'] <= metrics_end)].copy()
    filtered = filtered[filtered['Source'].str.lower() != 'inbound_direct']
    filtered = filtered[
        ~((filtered['Call_Status'].str.lower() == 'not done') & (filtered['Region'].str.lower() == 'africa')) &
        ~(filtered['Call_Status'].str.lower() == "can't happen")
    ]
    filtered['Cohort'] = filtered['Prop_Type'].fillna('').replace({
        'Good Prop': 'Good Prospect', 'Good prop': 'Good Prospect',
        'Raw Prop': 'Raw Prospect', 'Raw prop': 'Raw Prospect'
    })
    total_bookings = len(filtered)
    total_done = len(filtered[filtered['Call_Status'].str.lower() == 'done'])
    team_rate = round((total_done / total_bookings * 100) if total_bookings > 0 else 0, 1)
    good_data = filtered[filtered['Cohort'].str.lower() == 'good prospect']
    good_total = len(good_data)
    good_done = len(good_data[good_data['Call_Status'].str.lower() == 'done'])
    team_good_rate = round((good_done / good_total * 100) if good_total > 0 else 0, 1)

    snapshot['slides']['agent_attendance'] = {
        'agents': agent_attendance,
        'team_metrics': {
            'total_bookings': total_bookings,
            'total_done': total_done,
            'team_rate': team_rate,
            'good_total': good_total,
            'good_done': good_done,
            'team_good_rate': team_good_rate
        }
    }

    # Slide 2.1: Attendance Breakdown (metrics filter)
    snapshot['slides']['attendance_breakdown'] = {
        'breakdown': calc_attendance_breakdown(attendance_df, metrics_start, metrics_end),
        'in_touch_records': calc_in_touch_not_done(attendance_df, metrics_start, metrics_end)
    }

    # Slide 2.2: Response Rate Tracker (metrics filter)
    snapshot['slides']['response_tracker'] = {
        'metrics': calc_response_metrics(attendance_df, metrics_start, metrics_end),
        'daily': calc_daily_response(attendance_df, metrics_start, metrics_end),
        'agents': calc_agent_response(attendance_df, metrics_start, metrics_end),
        'model_accuracy': calc_model_accuracy(attendance_df, metrics_start, metrics_end),
        'blank_responses': calc_blank_response(attendance_df, metrics_start, metrics_end)
    }

    # Slide 3: Agent Sales (metrics filter)
    snapshot['slides']['agent_sales'] = calc_agent_sales(sales_df, metrics_start, metrics_end)

    # Slide 4: Bookings Trend (trends filter)
    snapshot['slides']['bookings_trend'] = {
        'outbound': calc_bookings_trend(attendance_df, trends_start, trends_end),
        'inbound': calc_inbound_bookings_trend(attendance_df, trends_start, trends_end)
    }

    # Slide 5: Attendance Trend (trends filter)
    snapshot['slides']['attendance_trend'] = {
        'attendance': calc_attendance_trend(attendance_df, trends_start, trends_end),
        'response_rate': calc_response_rate_trend(attendance_df, trends_start, trends_end),
        'overall_bookings': calc_overall_bookings_trend(attendance_df, trends_start, trends_end)
    }

    # Slide 6: Conversion Trend (trends filter)
    snapshot['slides']['conversion_trend'] = calc_conversion_trend(sales_df, trends_start, trends_end)

    # Slide 6.1: Conversion by Intent (trends filter)
    snapshot['slides']['conversion_by_intent'] = {
        'by_intent': calc_conversion_trend_by_intent(sales_df, trends_start, trends_end),
        'calls_low_intent': calc_calls_low_intent_trend(sales_df, trends_start, trends_end)
    }

    # Slide 7: Phone Call Analysis (metrics filter)
    snapshot['slides']['phone_calls'] = calc_phone_call_metrics(sales_df, metrics_start, metrics_end)

    # Slide 8: Source Breakdown (metrics filter)
    snapshot['slides']['source_breakdown'] = {
        'by_source': calc_source_breakdown(attendance_df, metrics_start, metrics_end),
        'by_source_agent': calc_source_agent_breakdown(attendance_df, metrics_start, metrics_end),
        'leakage': calc_leakage_metrics(attendance_df, metrics_start, metrics_end),
        'booking_source': calc_booking_source_breakdown(sales_df, metrics_start, metrics_end)
    }

    # Slide 9: Monthly Sales (trends filter)
    snapshot['slides']['monthly_sales'] = {
        'monthly': calc_monthly_sales(sales_df, trends_start, trends_end),
        'effective_attendance': calc_effective_attendance(sales_df, attendance_df, trends_start, trends_end)
    }

    # Slide 10: Marketing (static)
    snapshot['slides']['marketing'] = parse_marketing_data()

    # Daily Trends (if filters exist)
    if 'daily_trends_filter' in global_filters:
        daily_filters = global_filters['daily_trends_filter']

        # Calls & Purchases
        if 'calls_purchases' in daily_filters:
            cp_start = pd.to_datetime(daily_filters['calls_purchases']['start_date'])
            cp_end = pd.to_datetime(daily_filters['calls_purchases']['end_date']) + timedelta(days=1) - timedelta(seconds=1)
            calls_trend = calc_daily_calls_trend(sales_df, cp_start, cp_end)
            purchases_trend = calc_daily_purchases_trend(sales_df, cp_start, cp_end)
            total_calls = sum(d['calls_done'] for d in calls_trend)
            total_purchases = sum(d['purchases'] for d in purchases_trend)

            snapshot['slides']['daily_calls_purchases'] = {
                'filter': daily_filters['calls_purchases'],
                'calls_trend': calls_trend,
                'purchases_trend': purchases_trend,
                'summary': {
                    'total_calls': total_calls,
                    'total_purchases': total_purchases,
                    'avg_purchase_rate': round((total_purchases / total_calls * 100) if total_calls > 0 else 0, 1)
                }
            }

        # Attendance & Bookings
        if 'attendance_bookings' in daily_filters:
            ab_start = pd.to_datetime(daily_filters['attendance_bookings']['start_date'])
            ab_end = pd.to_datetime(daily_filters['attendance_bookings']['end_date']) + timedelta(days=1) - timedelta(seconds=1)
            attendance_trend = calc_daily_attendance_trend(attendance_df, ab_start, ab_end)
            bookings_trend = calc_daily_bookings_trend(attendance_df, ab_start, ab_end)
            total_bookings_daily = sum(d['total_bookings'] for d in bookings_trend)
            total_done_daily = sum(d['calls_done'] for d in attendance_trend)

            snapshot['slides']['daily_attendance_bookings'] = {
                'filter': daily_filters['attendance_bookings'],
                'attendance_trend': attendance_trend,
                'bookings_trend': bookings_trend,
                'summary': {
                    'total_bookings': total_bookings_daily,
                    'total_done': total_done_daily,
                    'avg_attendance_rate': round((total_done_daily / total_bookings_daily * 100) if total_bookings_daily > 0 else 0, 1)
                }
            }

    # Clean up any non-serializable objects (like Timestamp)
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(item) for item in obj]
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif pd.isna(obj):
            return None
        else:
            return obj

    snapshot = clean_for_json(snapshot)

    # Save to file
    snapshot_file = os.path.join(CONFIG_PATH, 'dashboard_snapshot.json')
    with open(snapshot_file, 'w', encoding='utf-8') as f:
        json.dump(snapshot, f, indent=2, default=str)

    return snapshot_file, snapshot

# Slide 10: Marketing
def parse_marketing_data():
    """Parse Marketing.txt file and extract metrics - matching Streamlit exactly"""
    try:
        marketing_file = os.path.join(CONFIG_PATH, 'Marketing.txt')
        with open(marketing_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Initialize metrics dictionary matching Streamlit structure
        metrics = {
            'revenue_current': 0,
            'revenue_target': 0,
            'revenue_red_alert': 0,
            'total_customers_current': 0,
            'total_customers_target': 0,
            'total_customers_red_alert': 0,
            'marketing_customers_current': 0,
            'marketing_customers_red_alert': 0,
            'sales_customers_current': 0,
            'sales_customers_red_alert': 0
        }

        # Parse the content line by line
        lines = content.split('\n')
        for line in lines:
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue

            # Parse key-value pairs
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()

                # Map keys to metrics dictionary
                try:
                    if key == 'REVENUE_CURRENT':
                        metrics['revenue_current'] = int(value)
                    elif key == 'REVENUE_TARGET':
                        metrics['revenue_target'] = int(value)
                    elif key == 'REVENUE_RED_ALERT':
                        metrics['revenue_red_alert'] = int(value)
                    elif key == 'TOTAL_CUSTOMERS_CURRENT':
                        metrics['total_customers_current'] = int(value)
                    elif key == 'TOTAL_CUSTOMERS_TARGET':
                        metrics['total_customers_target'] = int(value)
                    elif key == 'TOTAL_CUSTOMERS_RED_ALERT':
                        metrics['total_customers_red_alert'] = int(value)
                    elif key == 'MARKETING_CUSTOMERS_CURRENT':
                        metrics['marketing_customers_current'] = int(value)
                    elif key == 'MARKETING_CUSTOMERS_RED_ALERT':
                        metrics['marketing_customers_red_alert'] = int(value)
                    elif key == 'SALES_CUSTOMERS_CURRENT':
                        metrics['sales_customers_current'] = int(value)
                    elif key == 'SALES_CUSTOMERS_RED_ALERT':
                        metrics['sales_customers_red_alert'] = int(value)
                except ValueError:
                    pass

        return metrics
    except:
        return None

# =============================================================================
# ROUTES
# =============================================================================

@app.route('/')
def index():
    return redirect(url_for('dashboard', slide='1'))

@app.route('/save-filters', methods=['POST'])
def save_filters():
    """Save global filters and redirect back to current page"""
    try:
        filters = load_global_filters()

        # Update metrics filter
        filters['metrics_filter']['start_date'] = request.form.get('metrics_start_date', filters['metrics_filter']['start_date'])
        filters['metrics_filter']['end_date'] = request.form.get('metrics_end_date', filters['metrics_filter']['end_date'])

        # Update trends filter
        filters['trends_filter']['start_date'] = request.form.get('trends_start_date', filters['trends_filter']['start_date'])
        filters['trends_filter']['end_date'] = request.form.get('trends_end_date', filters['trends_filter']['end_date'])

        if save_global_filters(filters):
            flash('Filters applied and saved successfully!', 'success')
        else:
            flash('Error saving filters', 'error')

    except Exception as e:
        flash(f'Error: {str(e)}', 'error')

    # Redirect back to the current slide
    current_slide = request.form.get('current_slide', '1')
    return redirect(url_for('dashboard', slide=current_slide))

@app.route('/dashboard/<slide>')
def dashboard(slide):
    # Load global filters from persistent storage
    global_filters = load_global_filters()
    filter_type = get_filter_type_for_slide(slide)

    # Get the appropriate filter for this slide
    if slide in STATIC_SLIDES:
        # Slide 10 doesn't need date range - uses static Marketing.txt
        start_date = None
        end_date = None
        start = None
        end = None
    elif slide in TRENDS_SLIDES:
        # Use trends filter for trend slides
        start_date = global_filters['trends_filter']['start_date']
        end_date = global_filters['trends_filter']['end_date']
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date) + timedelta(days=1) - timedelta(seconds=1)
    else:
        # Use metrics filter for metrics slides
        start_date = global_filters['metrics_filter']['start_date']
        end_date = global_filters['metrics_filter']['end_date']
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date) + timedelta(days=1) - timedelta(seconds=1)

    # Load data
    try:
        sales_df = load_sales_data()
        attendance_df = load_attendance_data()
    except Exception as e:
        return render_template('error.html', error=str(e))

    # Calculate data based on slide
    data = {
        'slide': slide,
        'start_date': start_date,
        'end_date': end_date,
        'start_display': start.strftime('%b %d, %Y') if start else '',
        'end_display': end.strftime('%b %d, %Y') if end else '',
        # Global filter data for template
        'global_filters': global_filters,
        'filter_type': filter_type,
        'metrics_slides': METRICS_SLIDES,
        'trends_slides': TRENDS_SLIDES,
        'static_slides': STATIC_SLIDES
    }

    if slide == '1':
        data['metrics'] = calc_overview_metrics(sales_df, attendance_df, start, end)
        # Load comments
        try:
            with open(os.path.join(CONFIG_PATH, 'comments.txt'), 'r', encoding='utf-8') as f:
                data['comments'] = f.read()
        except:
            data['comments'] = None

    elif slide == '2':
        data['agents'] = calc_agent_attendance(attendance_df, start, end)
        # Calculate team-wide metrics - excluding Inbound_Direct
        filtered = attendance_df[(attendance_df['EventDate'] >= start) & (attendance_df['EventDate'] <= end)].copy()
        # Exclude Inbound_Direct
        filtered = filtered[filtered['Source'].str.lower() != 'inbound_direct']
        filtered = filtered[
            ~((filtered['Call_Status'].str.lower() == 'not done') & (filtered['Region'].str.lower() == 'africa')) &
            ~(filtered['Call_Status'].str.lower() == "can't happen")
        ]
        filtered['Cohort'] = filtered['Prop_Type'].fillna('').replace({
            'Good Prop': 'Good Prospect', 'Good prop': 'Good Prospect',
            'Raw Prop': 'Raw Prospect', 'Raw prop': 'Raw Prospect'
        })
        total_bookings = len(filtered)
        total_done = len(filtered[filtered['Call_Status'].str.lower() == 'done'])
        team_rate = round((total_done / total_bookings * 100) if total_bookings > 0 else 0, 1)
        good_data = filtered[filtered['Cohort'].str.lower() == 'good prospect']
        good_total = len(good_data)
        good_done = len(good_data[good_data['Call_Status'].str.lower() == 'done'])
        team_good_rate = round((good_done / good_total * 100) if good_total > 0 else 0, 1)
        data['team_rate'] = team_rate
        data['team_good_rate'] = team_good_rate
        data['total_bookings'] = total_bookings
        data['total_done'] = total_done
        data['good_total'] = good_total
        data['good_done'] = good_done

    elif slide == '2.1':
        data['breakdown'] = calc_attendance_breakdown(attendance_df, start, end)
        data['in_touch_records'] = calc_in_touch_not_done(attendance_df, start, end)

    elif slide == '2.2':
        data['metrics'] = calc_response_metrics(attendance_df, start, end)
        data['daily'] = calc_daily_response(attendance_df, start, end)
        data['agents'] = calc_agent_response(attendance_df, start, end)
        data['model'] = calc_model_accuracy(attendance_df, start, end)
        data['blank'] = calc_blank_response(attendance_df, start, end)

    elif slide == '3':
        data['agents'] = calc_agent_sales(sales_df, start, end)

    elif slide == '4':
        data['trend'] = calc_bookings_trend(attendance_df, start, end)
        data['inbound_trend'] = calc_inbound_bookings_trend(attendance_df, start, end)

    elif slide == '5':
        data['trend'] = calc_attendance_trend(attendance_df, start, end)
        data['response_trend'] = calc_response_rate_trend(attendance_df, start, end)
        data['overall_trend'] = calc_overall_bookings_trend(attendance_df, start, end)

    elif slide == '6':
        data['trend'] = calc_conversion_trend(sales_df, start, end)

    elif slide == '6.1':
        data['intent_trends'] = calc_conversion_trend_by_intent(sales_df, start, end)
        data['calls_low_intent'] = calc_calls_low_intent_trend(sales_df, start, end)

    elif slide == '7':
        data['agents'] = calc_phone_call_metrics(sales_df, start, end)

    elif slide == '8':
        data['source_metrics'] = calc_source_breakdown(attendance_df, start, end)
        data['source_agent_metrics'] = calc_source_agent_breakdown(attendance_df, start, end)
        data['leakage'] = calc_leakage_metrics(attendance_df, start, end)
        # Also need to calculate booking source metrics from sales data
        data['booking_source_metrics'] = calc_booking_source_breakdown(sales_df, start, end)

    elif slide == '9':
        data['monthly_metrics'] = calc_monthly_sales(sales_df, start, end)
        data['attendance_metrics'] = calc_effective_attendance(sales_df, attendance_df, start, end)

    elif slide == '10':
        data['marketing'] = parse_marketing_data()

    return render_template('dashboard.html', **data)

@app.route('/refresh-data')
def refresh_data():
    success, message = refresh_data_from_airtable()
    flash(message, 'success' if success else 'error')
    return redirect(request.referrer or url_for('index'))

# =============================================================================
# DAILY TRENDS ROUTES (Independent Filters)
# =============================================================================

@app.route('/daily-trends/<page>')
def daily_trends(page):
    """Daily Trends pages with independent filters"""
    global_filters = load_global_filters()

    # Ensure daily_trends_filter exists in the loaded filters
    if 'daily_trends_filter' not in global_filters:
        global_filters['daily_trends_filter'] = {
            'calls_purchases': {
                'start_date': (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                'end_date': datetime.now().strftime('%Y-%m-%d')
            },
            'attendance_bookings': {
                'start_date': (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                'end_date': datetime.now().strftime('%Y-%m-%d')
            }
        }

    # Load data
    try:
        sales_df = load_sales_data()
        attendance_df = load_attendance_data()
    except Exception as e:
        return render_template('error.html', error=str(e))

    data = {
        'page': page,
        'global_filters': global_filters
    }

    if page == '1':
        # Calls & Purchases Daily Trends
        filter_config = global_filters['daily_trends_filter']['calls_purchases']
        start_date = filter_config['start_date']
        end_date = filter_config['end_date']
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date) + timedelta(days=1) - timedelta(seconds=1)

        data['start_date'] = start_date
        data['end_date'] = end_date
        data['start_display'] = start.strftime('%b %d, %Y')
        data['end_display'] = end.strftime('%b %d, %Y')
        data['calls_trend'] = calc_daily_calls_trend(sales_df, start, end)
        data['purchases_trend'] = calc_daily_purchases_trend(sales_df, start, end)

        # Calculate totals
        total_calls = sum(d['calls_done'] for d in data['calls_trend'])
        total_purchases = sum(d['purchases'] for d in data['purchases_trend'])
        avg_rate = round((total_purchases / total_calls * 100) if total_calls > 0 else 0, 1)
        data['summary'] = {
            'total_calls': total_calls,
            'total_purchases': total_purchases,
            'avg_purchase_rate': avg_rate
        }

        return render_template('slides/daily_trends_calls_purchases.html', **data)

    elif page == '2':
        # Attendance & Bookings Daily Trends
        filter_config = global_filters['daily_trends_filter']['attendance_bookings']
        start_date = filter_config['start_date']
        end_date = filter_config['end_date']
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date) + timedelta(days=1) - timedelta(seconds=1)

        data['start_date'] = start_date
        data['end_date'] = end_date
        data['start_display'] = start.strftime('%b %d, %Y')
        data['end_display'] = end.strftime('%b %d, %Y')
        data['attendance_trend'] = calc_daily_attendance_trend(attendance_df, start, end)
        data['bookings_trend'] = calc_daily_bookings_trend(attendance_df, start, end)

        # Calculate totals
        total_bookings = sum(d['total_bookings'] for d in data['bookings_trend'])
        total_done = sum(d['calls_done'] for d in data['attendance_trend'])
        avg_rate = round((total_done / total_bookings * 100) if total_bookings > 0 else 0, 1)
        data['summary'] = {
            'total_bookings': total_bookings,
            'total_done': total_done,
            'avg_attendance_rate': avg_rate
        }

        return render_template('slides/daily_trends_attendance_bookings.html', **data)

    return redirect(url_for('daily_trends', page='1'))

@app.route('/save-daily-filters', methods=['POST'])
def save_daily_filters():
    """Save daily trends filters independently"""
    try:
        filters = load_global_filters()

        # Ensure daily_trends_filter exists
        if 'daily_trends_filter' not in filters:
            filters['daily_trends_filter'] = {
                'calls_purchases': {},
                'attendance_bookings': {}
            }

        page = request.form.get('page', '1')

        if page == '1':
            # Update calls_purchases filter
            filters['daily_trends_filter']['calls_purchases']['start_date'] = request.form.get('start_date')
            filters['daily_trends_filter']['calls_purchases']['end_date'] = request.form.get('end_date')
        elif page == '2':
            # Update attendance_bookings filter
            filters['daily_trends_filter']['attendance_bookings']['start_date'] = request.form.get('start_date')
            filters['daily_trends_filter']['attendance_bookings']['end_date'] = request.form.get('end_date')

        if save_global_filters(filters):
            flash('Filters applied and saved!', 'success')
        else:
            flash('Error saving filters', 'error')

    except Exception as e:
        flash(f'Error: {str(e)}', 'error')

    return redirect(url_for('daily_trends', page=page))

# =============================================================================
# EXPORT SNAPSHOT ROUTE
# =============================================================================

@app.route('/export-snapshot')
def export_snapshot():
    """Export current dashboard data to JSON file"""
    try:
        snapshot_file, snapshot = export_dashboard_snapshot()
        flash(f'Dashboard snapshot exported successfully!', 'success')

        # Return JSON response with file path
        return jsonify({
            'success': True,
            'message': 'Snapshot exported successfully',
            'file_path': snapshot_file,
            'exported_at': snapshot['exported_at'],
            'filters': snapshot['filters']
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/snapshot')
def get_snapshot():
    """API endpoint to get the latest snapshot data directly"""
    try:
        snapshot_file = os.path.join(CONFIG_PATH, 'dashboard_snapshot.json')
        if os.path.exists(snapshot_file):
            with open(snapshot_file, 'r', encoding='utf-8') as f:
                snapshot = json.load(f)
            return jsonify(snapshot)
        else:
            return jsonify({'error': 'No snapshot found. Visit /export-snapshot first.'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=5000)
