import argparse
import os
from azureml.core import Run

def one_hot_encoding(dataset, feature):
    dummy_dataset = pd.get_dummies(dataset[[feature]])
    encode_dataset = pd.concat([dataset, dummy_dataset], axis=1)
    encode_dataset = encode_dataset.drop([feature], axis=1)
    return(encode_dataset)

print("Transform some columns of the data")

run = Run.get_context()

filtered_data = run.input_datasets['normalized_data']
df = filtered_data.to_pandas_dataframe()

parser = argparse.ArgumentParser("transform")
parser.add_argument("--output_transform", type=str, help="transformed data")

args = parser.parse_args()

# +
# print("Argument 2(output final transformed data): %s" % args.output_transform)
# -

for col in ['age', 'height', 'weight', 'bmi','ap_hi', 'ap_lo', 'cardio']:
  df[col] = df[col].astype(float)

df['cholesterol']=df['cholesterol'].map({ 1: 'normal', 2: 'above normal', 3: 'well above normal'})
df['gluc']=df['gluc'].map({ 1: 'normal', 2: 'above normal', 3: 'well above normal'})
df['smoke']=df['smoke'].map({ 0: 'No', 1: 'Yes'})
df['alco']=df['alco'].map({ 0: 'No', 1: 'Yes'})
df['active']=df['active'].map({ 0: 'No', 1: 'Yes'})


for features in ['bmi','weight','ap_hi']:
    df[features]= boxcox1p(df[features],  boxcox_normmax(df[features] + 1))

encoding_features = ['cholesterol','gluc','smoke','alco','active']
for feature in encoding_features:
    df = one_hot_encoding(df, feature)

df.reset_index(inplace=True, drop=True)

# Writing the final dataframe to use for training in the following steps
if not (args.output_transform is None):
    os.makedirs(args.output_transform, exist_ok=True)
    print("%s created" % args.output_transform)
    path = args.output_transform + "/processed.parquet"
    write_df = df.to_parquet(path)
