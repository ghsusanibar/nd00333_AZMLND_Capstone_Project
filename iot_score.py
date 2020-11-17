import pickle
import json
import pandas
import joblib
from sklearn.linear_model import Ridge
from azureml.core.model import Model

def init():
    global model

    # The AZUREML_MODEL_DIR environment variable indicates
    # a directory containing the model file you registered.
    model_filename = 'model.pkl'
    print(model_filename)
    print(os.environ['AZUREML_MODEL_DIR'])
    model_path = os.path.join(os.environ['AZUREML_MODEL_DIR'], model_filename)

    model = joblib.load(model_path)

# note you can pass in multiple rows for scoring
def run(input_str):
    try:
        input_json = json.loads(input_str)
        input_df = pandas.DataFrame([[input_json['age'],input_json['height'],input_json['weight'],input_json['ap_hi'],input_json['ap_lo'],input_json['bmi'],input_json['cholesterol_above normal'],input_json['cholesterol_normal']
            ,input_json['cholesterol_well above normal'],input_json['gluc_above normal'],input_json['gluc_normal'],input_json['gluc_well above normal'],input_json['smoke_No'],input_json['smoke_Yes'],input_json['alco_No']
            ,input_json['alco_Yes'],input_json['active_No'],input_json['active_Yes']]])
        pred = model.predict(input_df)
        input_json['prediction']=pred[0]
        print("Prediction is ", pred[0])
    except Exception as e:
        input_json['prediction']=0.0
        result = str(e)  
        
    return [json.dumps(input_json)]