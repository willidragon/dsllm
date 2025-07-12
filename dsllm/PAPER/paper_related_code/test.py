import pandas as pd
from io import StringIO

# Load the raw CSV-like string into a DataFrame
raw_data = """class,precision,recall,f1-score,support,experiment
sleep,0.917,0.9463,0.9314,3946,baseline_100ds
sitting,0.8428,0.8142,0.8282,3121,baseline_100ds
standing,0.1,0.0175,0.0299,285,baseline_100ds
walking,0.2463,0.4809,0.3257,445,baseline_100ds
bicycling,0.7111,0.4638,0.5614,138,baseline_100ds
vehicle,0.3373,0.7446,0.4642,231,baseline_100ds
household-chores,0.412,0.5115,0.4564,741,baseline_100ds
manual-work,0.4,0.0081,0.0159,247,baseline_100ds
sports,0.1111,0.0217,0.0364,46,baseline_100ds
mixed-activity,0.2744,0.0893,0.1347,504,baseline_100ds
sleep,0.9444,0.9379,0.9411,3946,baseline_1000ds
sitting,0.806,0.703,0.751,3121,baseline_1000ds
standing,0.0782,0.2386,0.1177,285,baseline_1000ds
walking,0.2056,0.1978,0.2016,445,baseline_1000ds
bicycling,1.0,0.0145,0.0286,138,baseline_1000ds
vehicle,0.19,0.2641,0.221,231,baseline_1000ds
household-chores,0.3831,0.2389,0.2943,741,baseline_1000ds
manual-work,0.0,0.0,0.0,247,baseline_1000ds
sports,0.0,0.0,0.0,46,baseline_1000ds
mixed-activity,0.1697,0.3194,0.2216,504,baseline_1000ds
sleep,0.9425,0.9341,0.9383,3946,baseline_2000ds
sitting,0.645,0.8565,0.7359,3121,baseline_2000ds
standing,0.0217,0.0035,0.006,285,baseline_2000ds
walking,0.162,0.3236,0.2159,445,baseline_2000ds
bicycling,0.0,0.0,0.0,138,baseline_2000ds
vehicle,0.05,0.0216,0.0302,231,baseline_2000ds
household-chores,0.3241,0.1579,0.2123,741,baseline_2000ds
manual-work,0.0,0.0,0.0,247,baseline_2000ds
sports,0.0,0.0,0.0,46,baseline_2000ds
mixed-activity,0.131,0.0655,0.0873,504,baseline_2000ds
sleep,0.6955,0.929,0.7955,3946,lstm_1000ds
sitting,0.5695,0.5341,0.5513,3121,lstm_1000ds
standing,0.0,0.0,0.0,285,lstm_1000ds
walking,0.158,0.2876,0.204,445,lstm_1000ds
bicycling,0.0,0.0,0.0,138,lstm_1000ds
vehicle,0.0,0.0,0.0,231,lstm_1000ds
household-chores,0.483,0.0958,0.1599,741,lstm_1000ds
manual-work,0.0,0.0,0.0,247,lstm_1000ds
sports,0.0,0.0,0.0,46,lstm_1000ds
mixed-activity,0.159,0.1706,0.1646,504,lstm_1000ds
sleep,0.8636,0.9579,0.9083,3946,saits_1000ds
sitting,0.655,0.7616,0.7043,3121,saits_1000ds
standing,0.0892,0.0491,0.0633,285,saits_1000ds
walking,0.2134,0.3303,0.2593,445,saits_1000ds
bicycling,0.6739,0.2246,0.337,138,saits_1000ds
vehicle,0.4615,0.2597,0.3324,231,saits_1000ds
household-chores,0.3995,0.2308,0.2926,741,saits_1000ds
manual-work,0.0,0.0,0.0,247,saits_1000ds
sports,0.2381,0.1087,0.1493,46,saits_1000ds
mixed-activity,0.2311,0.1032,0.1427,504,saits_1000ds
sleep,0.798,0.9382,0.8624,3946,saits_2000ds
sitting,0.5417,0.6559,0.5933,3121,saits_2000ds
standing,0.0,0.0,0.0,285,saits_2000ds
walking,0.1429,0.0607,0.0852,445,saits_2000ds
bicycling,0.1333,0.0145,0.0261,138,saits_2000ds
vehicle,0.1,0.0087,0.0159,231,saits_2000ds
household-chores,0.2836,0.3954,0.3303,741,saits_2000ds
manual-work,0.0,0.0,0.0,247,saits_2000ds
sports,0.0,0.0,0.0,46,saits_2000ds
mixed-activity,0.1724,0.0099,0.0188,504,saits_2000ds
"""

df = pd.read_csv(StringIO(raw_data))

# Pivot the data for better LaTeX presentation: group by class and show metrics across experiments
pivot_df = df.pivot_table(index=["class"], columns="experiment", values=["precision", "recall", "f1-score"])

# Reset columns for LaTeX rendering
pivot_df.columns = ['{}_{}'.format(stat, exp) for stat, exp in pivot_df.columns]

# Reset index to include 'class' in the DataFrame
pivot_df.reset_index(inplace=True)

# Export to LaTeX
latex_table = pivot_df.to_latex(index=False, float_format="%.3f")

latex_table[:1500]  # Return a preview of the LaTeX output

# Export LaTeX table to a text file for easy copying
with open('research/dsllm/dsllm/PAPER/paper_related_code/latex_table_output.txt', 'w') as f:
    f.write(latex_table)