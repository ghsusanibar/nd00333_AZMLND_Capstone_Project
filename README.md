# Capstone Project

Table of Contents
=================
  * [Overview](#overview)
  * [Architecture](#architecture)
  * [Project Steps](#project-steps)
    + [Dataset](#dataset)
    + [AutoML Model](#automl-model)
      - [Pipeline](#pipeline)
      - [AutoML Config](#automl-config)
      - [RunDetails](#rundetails)
      - [Best Model](#best-model)
      - [Saving Model](#saving-model)
    + [Hyperdrive Model](#hyperdrive-model)
      - [Pipeline](#pipeline-1)
      - [HyperDrive config](#hyperdrive-config)
      - [RunDetails](#rundetails-1)
      - [Best Model](#best-model-1)
      - [Saving Model](#saving-model-1)
    + [Comparison of the two models](#comparison-of-the-two-models)
    + [Model Deployment](#model-deployment)
      - [Register Model](#register-model)
      - [Deploy Model](#deploy-model)
      - [Consume Model Endpoint](#consume-model-endpoint)
      - [Services cleanup](#services-cleanup)
    + [Standout Suggestions](#standout-suggestions)
      - [Convert model into ONNX format](#convert-model-into-onnx-format)
      - [Deploy model using IoT Edge](#deploy-model-using-iot-edge)
      - [Enable Logging](#enable-logging)
      - [Publish and Consume Pipeline](#publish-and-consume-pipeline)
  * [Future Improvements](#future-improvements)
  * [Screen Recording](#screen-recording)

## Overview
This is the Capstone Project of the Udacity Microsoft MLE Nanodegree.
In this project, we used all the knowledge obtained from this Nanodegree to build a Machine Learning Project. We had to get a dataset from an external resource and then build two experiments, one using the AutoML tool and the other one using Hyperdrive. Then we had to compare the perfomance of both models and deploy the best one. Also we had the oportunity to do some standout suggestions in order to take this project further.

## Architecture
First we have to choose a Dataset from an external resource like Kaggle, UCI, etc and import the dataset into the Azure ML Workspace. Then we have to train differents model using Automated ML and in aonther experiment train a model using Hyperdrive. After that we have to compare the performance of both best models and choose the best one in order to deploy it. Once deployed we have to test the model endpoint. At the end we can also do some Standout Suggestions such as convert the model into ONNX format and deploy the model using IoT Edge in order to demonstrate all the knowledge from this Nanodegree.

![architecture](/image/img100.jpg)

## Project Steps

### Dataset
In many regions with a high poverty rate there are hospitals with basic equipment. Sick people come to be treated for very common illnesses such as heart attacks, but many times doctors cannot help them due to lack of equipment. Thus this solution can help doctors to predict in time whether a person is prone to suffer an attack and thus give them timely treatment. Hope this solution can motivate to other people to use Machine Learning on Health. 
For that reason I chose the Cardiovascular Disease dataset from Kaggle. Cardiovascular Disease dataset is a Kaggle Dataset the containts history of health status of some persons. A group of them suffered a heart attackt. So using this dataset we can train a model in order to predict if a person could suffer a heart attack.
We can download the data from Kaggle page (https://www.kaggle.com/sulianova/cardiovascular-disease-dataset). Data required in order to predict if a person could suffer a heart attack: [age, gender, heigth, weight, blood pressure, cholesterol, glucosa, smole]. So I've download the data in the /kaggle directory and then I registered this Dataset in the Azure ML Studio.

![dataset](/image/img000.jpg)
![dataset](/image/img003.jpg)

### AutoML Model

#### Pipeline
As Data Scientist we know that before training a model, we have to do some process like feature cleaning and feature engineering in order to get better models. So for thaT reason I decided to build a Pipeline containg steps such as cleaning data, filtering, do some transformations and split the dataset into train and test sets. The last module correspond to the AutoML in order to train several kinds of models such as LightGBM, XGBoost, Logistic Regression, VotingEnsemble, among others algorithms.

#### AutoML Config
In order to run an AutoML experiment, we have to set up some parameters in the automl_config like the classification task, the primary metric, the label column, etc. In this case I chose the AUC_weithed as primary metric and specify the number of cross validation as 5. Then I created the AutoML step and I summitted the experiment. It took like 1 hour in order to run all the steps of the pipeline.

![automl](/image/img005.jpg)

#### RunDetails
I used the RunDetails tool in order to get some information about the AutoML experiment. We can see we got some information of the model like the accuracy and the AUC and also the status and description of the experiment.

![automl](/image/img010.jpg)

#### Best Model
After the experiment finished running we got different models trained, each one with its AUC metric. The best model was the VotingEnsemble with AUC=0.802. One advantege of the AutoML is that it also gives an explanation of the model. 

![automl](/image/img008.jpg)
![automl](/image/img007.jpg)

#### Saving Model
Once I got the best model of the AutoML experiment, I saved the model in the pickle format. Also I tested the model using the test dataset in order to be able of compare with other models.

### Hyperdrive Model

#### Pipeline
Similar to the previous experiment, I build a Pipeline containg steps such as cleaning data, filtering, do some transformations and split the dataset into train and test sets in order to do some feature engineering and help to get better models. The diffence is on the last module, which in this case is tha HyperDrive step.

#### HyperDrive config
In order to run a HyperDrive experiment we have to set up some previous details. First we have to specify the parameters to be tunned, in this case I chose the following parameters: num_leaves, max_depth, min_data_in_leaf and learning_rate. I defined the parameter space using random sampling.One of the The benefits of the random sampling is that the hyperparameter values are chosen from a set of discrete values or a distribution over a continuous range. So it tested several cases and not every combinations. It helped to reduce the time of hyperparameter tuning. Then I used the BanditPolicy as early stopping policy because it defines an early termination policy based on slack criteria, frequency and delay interval for evaluation. Any run that doesn't fall within the slack factor or slack amount of the evaluation metric with respect to the best performing run will be terminated. I wrote a training script in wich I use the LightGBM algorithm. Then I built an estimator that specifies the location of the script, sets up its fixed parameters, including the compute target and specifies the packages needed to run the script.

![hyperdrive](/image/img026.jpg)

Then I created the HyperDrive step using the hyperdrive_config and then I summited the experiment.

![hyperdrive](/image/img020.jpg)
![hyperdrive](/image/img022.jpg)

#### RunDetails
I used the RunDetails tool in order to get some information about the HyperDrive experiment. We can see a graphic of the AUC metric versus the runs and also the map of the parameters.

![hyperdrive](/image/img024.jpg)

#### Best Model
Once the HyperDrive experiment finished running we got different models trained, each one with its AUC metric and its hyperparameters. The best model was a LightGBM algorithm with AUC=0.8029 and hyperparameters learning_rate=0.016, max_depth=6, min_data_in_leaf=32 and num_leaves=16.

![hyperdrive](/image/img025.jpg)
![hyperdrive](/image/img027.jpg)

#### Saving Model
Once I got the best model of the HyperDrive experiment, I saved the model in the pickle format. Also I tested the model using the test dataset in order to be able of compare with the previous model.

### Comparison of the two models
For both experiments I used the AUC metric in order to be compare both models. We've seen the AUC in the validation set for the AutoML model was 0.8021, whereas for the Hyperdrive model was 0.8029. In addition I calculated the AUC in the test set, for the AutoML was 0.7317 and for the Hyperdrive model was 0.7978. So we see that the HyperDrive model is the best one between the two models. One reason to explain this is that the HyperDrive experiment focus on just one type of algorithm and try to find the best hyperparameters, in this case is the LightGBM which is an ensamble algorithm, whereas the AutoML tried differents algorithm, some of them basic algorithm like LogisticRegression. So now, we can deploy the best model.

![comparison](/image/img029.jpg)
![comparison](/image/img028.jpg)

### Model Deployment

#### Register Model
The firts step in order to deploy a model is register it. I used the register_model method from the best_run of the HyperDrive experiment. Then we can see that the model is registered.

![deployment](/image/img031.jpg)

#### Deploy Model
Beafore deploy the model, we have to create the scoring file and the environment file. Then we have to set up the parameters for the Azure Container Instance and then we can deploy the model.

![deployment](/image/img034.jpg)

The deployment process take some minutes but then we can see the information of the model deployed like the REST endpoint and the authentication keys.

![deployment](/image/img035.jpg)
![deployment](/image/img036.jpg)

#### Consume Model Endpoint
We can consumed the model endpoint using the HTTP API. First we have to specify the model endpoint and the primary key for authentication. Then we have to provide the data to predict in json format. With this information we make a request for the endpoint and we got the predictions.

![deployment](/image/img042.jpg)

#### Services cleanup
After all the steps, we can delete the ACI service and also we can delete the Compute cluster from its associated workspace in order to clean up services.

![deployment](/image/img052.jpg)

### Standout Suggestions

#### Convert model into ONNX format
The Open Neural Network Exchange (ONNX) is an open-sources portability platform for models that allows you to convert models from one framework to another, or even to deploy models to a device (such as an iOS or Android mobile device). I converted the best model into ONNX format usin the onnxtool.

![standout](/image/img044.jpg)
![standout](/image/img045.jpg)

#### Deploy model using IoT Edge
We can also deploy a model using the Azure IoT Edge. First I had to create an IoTHub service and an IoT Edge device. Then I created the iot scoring file, similar to the scoring file, and also I created the environment file. Then I created a docker image with the previous information. I created a deployment.json file that contains the modules to deploy to the device and the routes. Then I pushed this file to the IoT Hub, which will then send it to the IoT Edge device. The IoT Edge agent will then pull the Docker images and run them. So in this way the model is deployed in the iot edge device.

![iotedge](/image/img047.jpg)
![iotedge](/image/img048.jpg)
![iotedge](/image/img049.jpg)

#### Enable Logging
When we deploy a model, the Application Insight is not enable by default. So we can execute a couple of lines of code to enable it. After executed it we can see now the Application Insights is enabled and we can retrieve logs.

![logging](/image/img038.jpg)

In the Application Insights page we can see some information about the endpoint such as the server response time, the total server requets, the numer of failed requests, etc.

![logging](/image/img039.jpg)
![logging](/image/img040.jpg)

#### Publish and Consume Pipeline
As further step, I published the pipeline of the best model using the publish_pipeline method. It generated the Pipeline endpoint, in this case called "Cardio Pipeline" and in the portal we can see the REST endpoint and its status which is Active. 

![pipeline](/image/img050.jpg)

Finally, I consumed the pipeline endpoint and the Pipeline started to run again.

![pipeline](/image/img051.jpg)

## Future Improvements
We can improve this project in the future trying several options. For example in the AutoML experiment we can extend the training job time for the experiment and also we can specify the models which can be used for Experiments under Blocked model. In the Hyperdrive experiment, we can test another algorithms like XGBoost in order to get the best hyperparameters. Also we can add more steps for the pipeline, for example a step to do standarization and normalization of the variables. Finally I would recommend get an explanation of the model in order to explain the most important variables and also we can use the Fairness SDK to make an analysis if the model is getting bias for a certain variable like gender for example.

## Screen Recording
Finally I recorded a screencast that shows the entire process of the Capstone Project.

[![Screencast](https://img.youtube.com/vi/oQ2xcY-wr-w/0.jpg)](https://youtu.be/oQ2xcY-wr-w)
