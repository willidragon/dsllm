import pickle
import random
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.tsa.stattools import acf
import json
import re
import yaml

def get_prompt_dict(window_size_seconds):
    return {
        "trend_synonyms": {
            "upward": "downward",
            "ascending": "descending",
            "rising": "falling",
            "increasing": "decreasing"
        },
        "steady_synonyms": [
            "steady",
            "constant",
            "stable"
        ],
        "gen_summary_1": [
            f"The provided {{data_name}} are the {window_size_seconds}-second time-series recordings of three sensor channels.",
            f"The given {{data_name}} represent the {window_size_seconds}-second time-series data from three sensor channels.",
            f"{{data_name}} consist of {window_size_seconds}-second time-series measurements from three sensor channels.",
            f"{{data_name}} include {window_size_seconds}-second time-series readings from three sensor channels.",
            f"The supplied {{data_name}} are the {window_size_seconds}-second time-series values of three sensor channels.",
            f"The {window_size_seconds}-second time-series data of three sensor channels are provided in {{data_name}}.",
            f"Contained within {{data_name}} are the {window_size_seconds}-second time-series readings of three sensor channels.",
            f"The dataset {{data_name}} comprises {window_size_seconds}-second time-series observations from three sensor channels.",
            f"In {{data_name}}, you will find {window_size_seconds}-second time-series data recorded from three sensor channels.",
            f"{{data_name}} hold the {window_size_seconds}-second time-series measurements from three distinct sensor channels."
        ],
        "gen_summary_2": [
            "First, let's analyze the trend changes in each channel's data:\n",
            "To begin with, let's examine the trend variations in the data for each channel:\n",
            "Let's start by looking at the trend changes across each channel's data:\n",
            "Initially, let's analyze the trend shifts in the data for each channel:\n",
            "First of all, let's delve into the trend variations in each channel's data:\n"
        ],
        "gen_summary_3": [
            "The data exhibits {trend_num} distinct trends, with a total of {change_num} changes in trend observed.",
            "Analysis reveals {trend_num} separate trends within the data, undergoing a cumulative total of {change_num} shifts in direction.",
            "There are {trend_num} unique trends identified in the data, which altogether have shifted direction {change_num} times.",
            "The data outlines {trend_num} different patterns, with these patterns changing direction a total of {change_num} times.",
            "{trend_num} varied trends have been observed in the data, which altogether experienced {change_num} transitions."
        ],
        "gen_summary_4": [
            "Therefore, the data exhibited a {trend_type} trend for a cumulative period of {total_time} seconds",
            "In conclusion, the overall timespan of the data's {trend_type} tendency amounted to {total_time} seconds",
            "Summarizing the findings, the aggregate time during which the data displayed a {trend_type} pattern was {total_time} seconds",
            "The analysis reveals that the data's {trend_type} inclination persisted for a total of {total_time} seconds",
            "To encapsulate, the data's {trend_type} trend spanned a combined duration of {total_time} seconds"
        ],
        "gen_summary_5": [
            "a {trend_type} pattern for {total_time} seconds",
            "a {trend_type} trend for {total_time} seconds",
            "a {trend_type} pattern for a total of {total_time} seconds",
            "a {trend_type} trend for a total of {total_time} seconds",
            "a {trend_type} pattern for a sum of {total_time} seconds"
        ],
        "gen_summary_6": [
            "The overall trend is {overall_trend}.",
            "The general trend observed is {overall_trend}.",
            "Overall, the trend is {overall_trend}.",
            "The primary trend detected is {overall_trend}.",
            "In summary, the overall trend is {overall_trend}."
        ],
        "conclude": [
            "Therefore, the human activity represented by the given data should be {activity}.",
            "Hence, the human activity indicated by the provided data should be {activity}.",
            "Thus, the human activity shown by the given data is likely to be {activity}.",
            "As a result, the human activity reflected in the provided data should be {activity}.",
            "Consequently, the human activity depicted by the given data should be {activity}."
        ]
    }

Q_TEMPLATES = [
    "Which human activity does this {data_name} segment, consisting of {channel_num} channels, represent?",
    "What human activity is captured in this {data_name} segment with {channel_num} channels?",
    "Which human action is depicted in this {channel_num}-channel {data_name} segment?",
    "Can you identify the human activity represented in this {channel_num}-channel {data_name} segment?",
    "What human behavior is showcased in this {data_name} that includes {channel_num} channels?"
]

# Activity mapping for Capture24
ACTIVITIES = {
    0: "sleep",
    1: "sitting",
    2: "standing",
    3: "walking",
    4: "bicycling",
    5: "vehicle",
    6: "household-chores",
    7: "manual-work",
    8: "sports",
    9: "mixed-activity"
}

def num_to_words(num):
    units = ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    teens = ['ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen']
    tens = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']
    scales = ['', 'thousand', 'million', 'billion']

    if num < 0:
        return "minus " + num_to_words(abs(num))
    if num < 10:
        return units[int(num)]
    if num < 20:
        return teens[int(num) - 10]
    if num < 100:
        return tens[int(num) // 10] + (" " + num_to_words(num % 10) if num % 10 != 0 else "")
    if num < 1000:
        return units[int(num) // 100] + " hundred" + (" " + num_to_words(num % 100) if num % 100 != 0 else "")
    for i, scale in enumerate(scales[1:], 1):
        if num < 1000 ** (i + 1):
            return num_to_words(num // (1000 ** i)) + " " + scale + (" " + num_to_words(num % (1000 ** i)) if num % (1000 ** i) != 0 else "")

def convert_number(num):
    if '.' in str(num):
        whole, decimal = str(num).split('.')
        if decimal == '0':
            return num_to_words(int(num))
        else:
            return num_to_words(int(whole)) + " point " + " ".join([num_to_words(int(digit)) for digit in decimal])
    else:
        return num_to_words(int(num))

def capitalize_first_letter(string):
    if len(string) == 0:
        return string
    else:
        return string[0].upper() + string[1:]

def check_a_an(sentence):
    words = re.findall(r'\b\w+\b', sentence)
    vowels = 'aeiouAEIOU'
    corrected_sentence = sentence

    for i in range(len(words)):
        if words[i] in ['a', 'an', 'A', 'An']:
            if i + 1 < len(words):
                next_word = words[i + 1]
                if words[i] == 'a' and next_word[0] in vowels:
                    corrected_sentence = corrected_sentence.replace(f' a {next_word}', f' an {next_word}', 1)
                elif words[i] == 'A' and next_word[0] in vowels:
                    corrected_sentence = corrected_sentence.replace(f' A {next_word}', f' An {next_word}', 1)
                elif words[i] == 'an' and next_word[0] not in vowels:
                    corrected_sentence = corrected_sentence.replace(f' an {next_word}', f' a {next_word}', 1)
                elif words[i] == 'An' and next_word[0] not in vowels:
                    corrected_sentence = corrected_sentence.replace(f' An {next_word}', f' A {next_word}', 1)

    return corrected_sentence

def analyze_trend(time_series, sample_rate, start_point=0):
    """
    Analyze the trend of time series data.

    Parameters:
    - time_series (list): A list of time series data points.
    - sample_rate (float): The sampling rate of the data (Hz)
    - start_point (int): The starting point in the time series

    Returns:
    - DataFrame: A DataFrame with columns: from_time, to_time, from_value, to_value, trend.
    """
    time_interval = 1 / sample_rate
    from_time, to_time, from_value, to_value, trend = [], [], [], [], []

    for i in range(len(time_series) - 1):
        start_time = round((start_point + i) * time_interval, 2)
        end_time = round((start_point + i + 1) * time_interval, 2)
        start_val = time_series[i]
        end_val = time_series[i + 1]

        if start_val == end_val:
            trend_type = 'steady'
        elif start_val < end_val:
            trend_type = 'increase'
        else:
            trend_type = 'decrease'

        from_time.append(start_time)
        to_time.append(end_time)
        from_value.append(start_val)
        to_value.append(end_val)
        trend.append(trend_type)

    result_df = pd.DataFrame({
        'from_time': from_time,
        'to_time': to_time,
        'from_value': from_value,
        'to_value': to_value,
        'trend': trend
    })

    return result_df

def merge_adjacent_rows(df):
    """
    Merge adjacent rows with the same trend into a new dataframe.
    """
    merged_rows = []
    current_start_time = df.iloc[0]['from_time']
    current_start_value = df.iloc[0]['from_value']
    current_trend = df.iloc[0]['trend']
    current_values = [current_start_value]

    for index, row in df.iterrows():
        if row['trend'] == current_trend:
            current_values.append(row['to_value'])
        else:
            merged_rows.append({
                'start_time': current_start_time,
                'end_time': df.iloc[index - 1]['to_time'],
                'start_value': current_start_value,
                'end_value': df.iloc[index - 1]['to_value'],
                'trend': current_trend,
                'values': current_values.copy()
            })
            current_start_time = row['from_time']
            current_start_value = row['from_value']
            current_trend = row['trend']
            current_values = [current_start_value, row['to_value']]

    merged_rows.append({
        'start_time': current_start_time,
        'end_time': df.iloc[-1]['to_time'],
        'start_value': current_start_value,
        'end_value': df.iloc[-1]['to_value'],
        'trend': current_trend,
        'values': current_values
    })

    return pd.DataFrame(merged_rows)

def calculate_total_time(df):
    """
    Calculate the total duration for each trend in the dataframe.
    """
    total_time_by_trend = df.groupby('trend').apply(
        lambda x: round((x['end_time'] - x['start_time']).sum(), 2)).reset_index(
        name='total_time')
    return total_time_by_trend

def format_float_2_int(num):
    if isinstance(num, float) and num.is_integer():
        return int(num)
    else:
        return num

def select_random_pair(window_size_seconds):
    word_pairs = get_prompt_dict(window_size_seconds)["trend_synonyms"]
    upward_word = random.choice(list(word_pairs.keys()))
    downward_word = word_pairs[upward_word]
    steady_word = random.choice(get_prompt_dict(window_size_seconds)["steady_synonyms"])
    return [upward_word, downward_word, steady_word]

def choose_word(input_trend, pair):
    if input_trend == "steady":
        return pair[2]
    elif input_trend == "increase" or input_trend == "upward":
        return pair[0]
    else:
        return pair[1]

def choose_decimal_places(std_dev):
    if std_dev < 0.01:
        return 6
    elif std_dev < 0.1:
        return 4
    elif std_dev < 1:
        return 3
    else:
        return 2

def generate_correlation_text(correlation_df):
    text = "Pearson Correlation Matrix for each channel:\n"
    for row in correlation_df.index:
        for col in correlation_df.columns:
            if row < col:
                correlation_value = correlation_df.loc[row, col]
                if correlation_value >= 0.7:
                    correlation_description = "strongly positively correlated"
                elif correlation_value >= 0.3:
                    correlation_description = "moderately positively correlated"
                elif correlation_value >= 0.1:
                    correlation_description = "weakly positively correlated"
                elif correlation_value <= -0.7:
                    correlation_description = "strongly negatively correlated"
                elif correlation_value <= -0.3:
                    correlation_description = "moderately negatively correlated"
                elif correlation_value <= -0.1:
                    correlation_description = "weakly negatively correlated"
                else:
                    correlation_description = "not significantly correlated"

                text += f"The correlation between {row} and {col} is {correlation_description}.\n"
    return text

def round_to_8_decimals(number):
    return f'{number:.8f}'.rstrip('0').rstrip('.')

def gen_reason(d, pair_list, data_type, sampling_rate, window_size_seconds):
    """
    Generate descriptive analysis of Capture24 sensor data.
    Parameters:
    - d: numpy array of shape (window_size, 3) containing x, y, z accelerometer data
    - pair_list: list of trend description words
    - data_type: string describing the type of data
    - sampling_rate: the actual (downsampled) sampling rate to use for time calculations
    - window_size_seconds: the size of the time window in seconds
    Returns:
    - Dictionary containing summary text, trend analysis, and correlation analysis
    """
    prompt_dict = get_prompt_dict(window_size_seconds)
    assert len(d[0]) == 3  # Capture24 has 3 channels (x, y, z)
    # Extract channels
    acc_x = d[:, 0]
    acc_y = d[:, 1]
    acc_z = d[:, 2]
    reading_list = [acc_x, acc_y, acc_z]
    reading_name = ["x-axis accelerometer", "y-axis accelerometer", "z-axis accelerometer"]
    info = {
        reading_name[0]: {},
        reading_name[1]: {},
        reading_name[2]: {}
    }
    smry_text = []
    trend_text = []
    corr_text = []
    smry_text.append("Statistics for each channel:\n")
    sr = sampling_rate  # Use the actual (downsampled) sampling rate
    for r, n in zip(reading_list, reading_name):
        data_df = merge_adjacent_rows(analyze_trend(r, sr))
        total_time_df = calculate_total_time(data_df)
        trend_text.append(n + ": ")
        info[n]["trend_num"] = len(total_time_df)
        info[n]["total_change_num"] = len(data_df)
        selected_template3 = random.choice(prompt_dict["gen_summary_3"])
        selected_template4 = random.choice(prompt_dict["gen_summary_4"])
        selected_template5 = random.choice(prompt_dict["gen_summary_5"])
        trend_text.append(capitalize_first_letter(
            selected_template3.format(
                trend_num=random.choice([info[n]["trend_num"], convert_number(info[n]["trend_num"])]),
                change_num=random.choice([info[n]["total_change_num"], convert_number(info[n]["total_change_num"])])
            )))
        if 'trend_total_time' not in info[n]:
            info[n]['trend_total_time'] = {}
        for index, t in total_time_df.iterrows():
            info[n]["trend_total_time"][t["trend"]] = t['total_time']
        i_t = 0
        for index, t in total_time_df.iterrows():
            if i_t == 0:
                if len(total_time_df) == 1:
                    trend_text.append(capitalize_first_letter(
                        selected_template4.format(
                            trend_type=choose_word(t["trend"], pair_list),
                            total_time=f"{t['total_time']:.2f}"
                        )) + ".")
                else:
                    trend_text.append(capitalize_first_letter(
                        selected_template4.format(
                            trend_type=choose_word(t["trend"], pair_list),
                            total_time=f"{t['total_time']:.2f}"
                        )) + ",")
            elif i_t < len(total_time_df) - 1:
                trend_text.append(
                    selected_template5.format(
                        trend_type=choose_word(t["trend"], pair_list),
                        total_time=f"{t['total_time']:.2f}"
                    ) + ",")
            else:
                trend_text.append(
                    "and " + selected_template5.format(
                        trend_type=choose_word(t["trend"], pair_list),
                        total_time=f"{t['total_time']:.2f}"
                    ) + ".")
            i_t += 1
        differences = np.diff(r)
        sum_of_differences = np.sum(differences)
        if sum_of_differences > 0:
            info[n]["overall_trend"] = "upward"
        elif sum_of_differences < 0:
            info[n]["overall_trend"] = "downward"
        else:
            info[n]["overall_trend"] = "steady"
        if info[n]["total_change_num"] > 1:
            selected_template7 = random.choice(prompt_dict["gen_summary_6"])
            trend_text.append(capitalize_first_letter(
                selected_template7.format(overall_trend=info[n]["overall_trend"])) + '\n')
        info[n]["min"] = np.min(r)
        info[n]["max"] = np.max(r)
        info[n]["median"] = np.median(r)
        info[n]["mean"] = np.mean(r)
        info[n]["std_dev"] = np.std(r)
        decimal_places = choose_decimal_places(info[n]["std_dev"])
        smry_text.append(f"{n}: Mean={round_to_8_decimals(info[n]['mean'])}, StdDev={round_to_8_decimals(info[n]['std_dev'])}\n")
        trend_counts = data_df['trend'].value_counts()
        if 'trend_total_changes' not in info[n]:
            info[n]['trend_total_changes'] = {}
        for i_n in range(len(trend_counts)):
            info[n]["trend_total_changes"][trend_counts.index[i_n]] = trend_counts.values[i_n]
    correlation_matrix = np.corrcoef(np.array(reading_list).T, rowvar=False)
    correlation_df = pd.DataFrame(correlation_matrix, columns=reading_name, index=reading_name)
    corr_text.append(generate_correlation_text(correlation_df))
    return {
        'smry_text': ' '.join(smry_text),
        'trend_text': ' '.join(trend_text),
        'corr_text': ' '.join(corr_text)
    }

def QA_gen(label_dict, data, pair_list, window_size_seconds, sampling_rate):
    """
    Generate question and answer for Capture24 data segment.
    Parameters:
    - label_dict: dictionary containing activity label information
    - data: numpy array of shape (window_size, 3) containing x, y, z accelerometer data
    - pair_list: list of trend description words
    - window_size_seconds: the size of the time window in seconds
    - sampling_rate: the actual (downsampled) sampling rate to use for time calculations
    Returns:
    - Dictionary containing question, summary statistics, trend analysis, correlation analysis, and ground truth answer
    """
    channel_num = data.shape[1]
    selected_template = random.choice(Q_TEMPLATES)
    data_types = ["time series data", "sensor data", "normalized time series data", "normalized sensor data"]
    selected_data_type = random.choice(data_types)
    c = random.choice([channel_num, convert_number(channel_num)])
    question = selected_template.format(data_name=selected_data_type, channel_num=c)
    reason = gen_reason(data, pair_list, selected_data_type, sampling_rate, window_size_seconds)
    # Robustly extract activity label
    if 'activity' in label_dict:
        gt = ACTIVITIES[int(label_dict['activity'])]
    elif 'activity_category' in label_dict:
        # Assume 1-based, so subtract 1
        gt = ACTIVITIES[int(label_dict['activity_category']) - 1]
    elif 'activity_name' in label_dict:
        activity_name = label_dict['activity_name']
        activity_idx = {v: k for k, v in ACTIVITIES.items()}[activity_name]
        gt = ACTIVITIES[activity_idx]
    else:
        raise KeyError(f"No activity label found in label_dict: {label_dict}")
    return {
        "Q": question,
        "smry": reason['smry_text'],
        "trend_text": reason['trend_text'],
        "corr_text": reason['corr_text'],
        "A": gt
    }