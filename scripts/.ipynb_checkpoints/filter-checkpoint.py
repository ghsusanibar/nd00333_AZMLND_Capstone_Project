import argparse
import os
from azureml.core import Run

print("Filtering some outliers")

run = Run.get_context()

cleaned_data = run.input_datasets["cleaned_data"]
df = cleaned_data.to_pandas_dataframe()

parser = argparse.ArgumentParser("filter")
parser.add_argument("--output_filter", type=str, help="filter some outliers")

args = parser.parse_args()

# +
# print("Argument (output filtered data path): %s" % args.output_filter)
# -

for col in ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'cardio']:
  df[col] = df[col].astype(float)

df = df[(df['height']<225) & (df['weight']>20.0)]
df["bmi"] = (df["weight"]/ (df["height"]/100)**2).round(1)
df = df[(df['bmi']>10) & (df['bmi']<100)]
df.age = np.round(df.age/365.25,decimals=1)
df= df[(df['ap_lo']<360) & (df['ap_hi']<360)].copy()
df= df[(df['ap_lo']>20) & (df['ap_hi']>20)].copy()
df=df[df['ap_hi']>df['ap_lo']]

df.reset_index(inplace=True, drop=True)

if not (args.output_filter is None):
    os.makedirs(args.output_filter, exist_ok=True)
    print("%s created" % args.output_filter)
    path = args.output_filter + "/processed.parquet"
    write_df = df.to_parquet(path)
