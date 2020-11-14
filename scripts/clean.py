import argparse
import os
from azureml.core import Run

print("Cleans the input data")

run = Run.get_context()
raw_data = run.input_datasets["raw_data"]

new_df = raw_data.to_pandas_dataframe()

parser = argparse.ArgumentParser(description='clean')
parser.add_argument("--output_clean", type=str, help="cleaned data directory")
parser.add_argument("--useful_columns", type=str, help="useful columns to keep")

args = parser.parse_args()

# +
# print("Argument 1(columns to keep): %s" % str(args.useful_columns.strip("[]").split(";")))
# print("Argument 2(output cleaned data path): %s" % args.output_clean)
# -

useful_columns = [s.strip().strip("'") for s in args.useful_columns.strip("[]").split("\;")]

new_df = (raw_data.to_pandas_dataframe().dropna(how='all'))[useful_columns]

new_df.reset_index(inplace=True, drop=True)

if not (args.output_clean is None):
    os.makedirs(args.output_clean, exist_ok=True)
    print("%s created" % args.output_clean)
    path = args.output_clean + "/processed.parquet"
    write_df = new_df.to_parquet(path)
