from doctest import FAIL_FAST
from unicodedata import category
import numpy as np
import pandas as pd
import tensorflow as tf                
from statistics import mode
from pandas.api.types import CategoricalDtype

class HDD():
    
    def __init__(self,channel=1):
        self.channel  = channel
    



    def bpredict(age_input, sex_input, cp_input, BO_input, ECG_input, HR_input):
        age_input = 20
        sex_input = 0
        cp_input = 1
        BO_input = 1
        ECG_input = 1

        tf.logging.set_verbosity(tf.logging.ERROR)
        pd.options.mode.chained_assignment = None
        np.random.seed(100)

        data_min = pd.Series([0.0,94.0,126.0,71.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], index=['age','resting_blood_pressure','cholesterol','max_heart_rate_achieved','st_depression','num_major_vessels','target','sex_female','chest_pain_type_atypical angina','chest_pain_type_non-anginal pain','chest_pain_type_typical angina','fasting_blood_sugar_lower than 120mg/ml','rest_ecg_left ventricular hypertrophy','rest_ecg_normal','exercise_induced_angina_yes','st_slope_flat','st_slope_upsloping','thalassemia_normal','thalassemia_reversible defect'])
        data_max = pd.Series([100.0,200.0,564.0,202.0,6.2,4.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0], index=['age','resting_blood_pressure','cholesterol','max_heart_rate_achieved','st_depression','num_major_vessels','target','sex_female','chest_pain_type_atypical angina','chest_pain_type_non-anginal pain','chest_pain_type_typical angina','fasting_blood_sugar_lower than 120mg/ml','rest_ecg_left ventricular hypertrophy','rest_ecg_normal','exercise_induced_angina_yes','st_slope_flat','st_slope_upsloping','thalassemia_normal','thalassemia_reversible defect'])

        myData = pd.read_csv("test.csv",encoding='utf-8')
        #myData = pd.DataFrame(np.array([age_input,sex_input,cp_input,BO_input,200.0,1.0,ECG_input,150.0,0.0,2.3,0.0,0.0,0.0,1.0]),index=[0],columns=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target'])
      
        myData['age'] = age_input
        myData['sex'] = sex_input
        myData['cp'] = cp_input
        myData['trestbps'] = BO_input
        myData['restecg'] = ECG_input

      

        myData.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved', 'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']

        myData['sex'][myData['sex'] == 0] = 'female'
        myData['sex'][myData['sex'] == 1] = 'male'
        myData['sex'] = myData['sex'].astype(CategoricalDtype( categories=['male','female']))
        myData['chest_pain_type'][myData['chest_pain_type'] == 1] = 'typical angina'
        myData['chest_pain_type'][myData['chest_pain_type'] == 2] = 'atypical angina'
        myData['chest_pain_type'][myData['chest_pain_type'] == 3] = 'non-anginal pain'
        myData['chest_pain_type'][myData['chest_pain_type'] == 4] = 'asymptomatic'
        myData['chest_pain_type'] = myData['chest_pain_type'].astype(CategoricalDtype(categories=['asymptomatic','atypical angina','non-anginal pain','typical angina']))
        myData['fasting_blood_sugar'][myData['fasting_blood_sugar'] == 0] = 'lower than 120mg/ml'
        myData['fasting_blood_sugar'][myData['fasting_blood_sugar'] == 1] = 'greater than 120mg/ml'
        myData['fasting_blood_sugar'] = myData['fasting_blood_sugar'].astype(CategoricalDtype(categories=['greater than 120mg/ml','lower than 120mg/ml']))
        myData['rest_ecg'][myData['rest_ecg'] == 0] = 'normal'
        myData['rest_ecg'][myData['rest_ecg'] == 1] = 'ST-T wave abnormality'
        myData['rest_ecg'][myData['rest_ecg'] == 2] = 'left ventricular hypertrophy'
        myData['rest_ecg'] = myData['rest_ecg'].astype(CategoricalDtype( categories=['ST-T wave abnormality','left ventricular hypertrophy','normal']))
        myData['exercise_induced_angina'][myData['exercise_induced_angina'] == 0] = 'no'
        myData['exercise_induced_angina'][myData['exercise_induced_angina'] == 1] = 'yes'
        myData['exercise_induced_angina'] = myData['exercise_induced_angina'].astype(CategoricalDtype( categories=['no','yes']))
        myData['st_slope'][myData['st_slope'] == 1] = 'upsloping'
        myData['st_slope'][myData['st_slope'] == 2] = 'flat'
        myData['st_slope'][myData['st_slope'] == 3] = 'downsloping'
        myData['st_slope'] = myData['st_slope'].astype(CategoricalDtype(categories=['downsloping','flat','upsloping',]))
        myData['thalassemia'][myData['thalassemia'] == 0] = 'mystory'
        myData['thalassemia'][myData['thalassemia'] == 1] = 'normal'
        myData['thalassemia'][myData['thalassemia'] == 2] = 'fixed defect'
        myData['thalassemia'][myData['thalassemia'] == 3] = 'reversible defect'
        myData['thalassemia'] = myData['thalassemia'].astype(CategoricalDtype( categories=['mystory','fixed defect','normal','reversible defect']))

        myData = pd.get_dummies(myData, drop_first=True)
       

        myData = (myData - data_min) / (data_max - data_min)
        # x_train, x_test, y_train, y_test = train_test_split(myData.drop('target', axis=1),
        #                                                     myData['target'], test_size=1, random_state=0)
     
        x_test = myData.drop('target',axis=1)

        nn_model = tf.keras.models.load_model(r'heart_disease_detect.h5')
        y_predicted = (nn_model.predict(x_test) > 0.5)
        if y_predicted == False:
            result = 0
        else:
            result = 1
        #y_predicted = nn_model.predict(x_test)
        return result

