import joblib
import numpy as np
import pandas as pd
import os

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType

from azureml.core.model import Model
import logging
logging.basicConfig(level=logging.DEBUG)
print(Model.get_model_path(model_name='Cardio-hd-Model'))


# The init() method is called once, when the web service starts up.
#
# Typically you would deserialize the model file, as shown here using joblib,
# and store it in a global variable so your run() method can access it later.
def init():
    global model

    # The AZUREML_MODEL_DIR environment variable indicates
    # a directory containing the model file you registered.
    model_filename = 'model.pkl'
    print(model_filename)
    print(os.environ['AZUREML_MODEL_DIR'])
    model_path = os.path.join(os.environ['AZUREML_MODEL_DIR'], model_filename)

    model = joblib.load(model_path)


# +
# The run() method is called each time a request is made to the scoring API.
#
# Shown here are the optional input_schema and output_schema decorators
# from the inference-schema pip package. Using these decorators on your
# run() method parses and validates the incoming payload against
# the example input you provide here. This will also generate a Swagger
# API document for your web service.

input_sample = pd.DataFrame({"age": pd.Series([0.0], dtype="float64"), "height": pd.Series([0.0], dtype="float64"), "weight": pd.Series([0.0], dtype="float64"), "ap_hi": pd.Series([0.0], dtype="float64"), "ap_lo": pd.Series([0.0], dtype="float64"), "bmi": pd.Series([0.0], dtype="float64"), "cholesterol_above normal": pd.Series([0.0], dtype="float64"), "cholesterol_normal": pd.Series([0.0], dtype="float64"), "cholesterol_well above normal": pd.Series([0.0], dtype="float64"), "gluc_above normal": pd.Series([0.0], dtype="float64"), "gluc_normal": pd.Series([0.0], dtype="float64"), "gluc_well above normal": pd.Series([0.0], dtype="float64"), "smoke_No": pd.Series([0.0], dtype="float64"), "smoke_Yes": pd.Series([0.0], dtype="float64"), "alco_No": pd.Series([0.0], dtype="float64"), "alco_Yes": pd.Series([0.0], dtype="float64"), "active_No": pd.Series([0.0], dtype="float64"), "active_Yes": pd.Series([0.0], dtype="float64")})
output_sample = np.array([0.4429])
@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    # Use the model object loaded by init().
    result = model.predict(data)

    # You can return any JSON-serializable object.
    return result.tolist()
# -


