import streamlit as st 
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
#st.markdown(
#    """
#    <style>
#    .stApp {
#        background-color: #ffffff;  /* Replace with your desired color */
#    }
#    </style>
#    """,
#    unsafe_allow_html=True
#)
st.set_page_config(layout="wide")
st.title('Веб-сервис для прогнозирования гипоксии плода')
stage = 0
# data of the patient
col1, col2 = st.columns(2)
with col1:
    name = st.text_input("ФИО пациентки - пока здесь вписываем id", 0)
with col2:
    oms = st.text_input("ОМС пациентки - пока здесь вписываем regular или hypoxia", "regular")  
stage = 1

if stage > 0:
    # read corresponding csv
    csv_name = oms + '_added.csv'
    df = pd.read_csv(csv_name)
    patient_df = df.loc[df['folder'] == int(name)].reset_index(drop = True)
    if patient_df.empty:
        st.write('Проверьте правильность написания ФИО и ОМС')
    else:
        with col1:
            st.subheader('Результаты анализа крови')
            if pd.notna(patient_df.loc[0,'gases_ph']):
                df_show = pd.DataFrame({'Анализ': ['Ph', 'Co2', 'Glucosa', 'Lactat', 'Be'],
                                        'Результаты': [np.array(patient_df['gases_ph'])[0], 
                                                    np.array(patient_df['gases_co2'])[0],
                                                    np.array(patient_df['glucosa'])[0],
                                                    np.array(patient_df['lactat'])[0],
                                                    np.array(patient_df['be'])[0]
                                                    ]})
                st.write(df_show)
            else:
                st.write('Данные анализов отсутствуют')
        with col2:
            st.subheader('Показатели КТГ')
            if pd.notna(patient_df.loc[0,'gases_ph']):
                df_show = pd.DataFrame({'Показатели': ['Ритм', 'Вариабельность', 
                                                    'Процент частот ниже 120', 
                                                    'Процент частот ниже 100', 
                                                    'Процент частот выше 160',
                                                    'Процент частот выше 180'],
                                        'Результаты': [np.array(patient_df['bpm_rythm'])[0], 
                                                    np.array(patient_df['bpm_variability'])[0],
                                                    np.array(patient_df['bpm_120'])[0] * 100,
                                                    np.array(patient_df['bpm_100'])[0] * 100,
                                                    np.array(patient_df['bpm_160'])[0] * 100,
                                                    np.array(patient_df['bpm_180'])[0] * 100
                                                    ]})
                st.write(df_show)
            else:
                st.write('Данные КТГ отсутствуют')
    
    
    
        model = CatBoostClassifier() 
        model.load_model('hypoxia_model.cbm')
    
        test_df = patient_df[['gases_ph', 'glucosa', 'lactat', 'be', 'ph_dif', 'glu_dif', 'lac_dif', 
                            'be_dif', 'bpm_rythm', 'bpm_variability', 'bpm_25', 'bpm_50', 'bpm_75', 
                            'bpm_160', 'bpm_180', 'bpm_100']]
        res = model.predict_proba(test_df)[0][1] * 100
        st.subheader(f'Вероятность развития гипоксии: {res:.2f}%')
        if (res < 30):
            short = 'низкая'
            long = 'низкая'
        elif (res >= 30) and (res < 60):
            short = 'средняя'
            long = 'средняя'
        else:
            short = 'высокая'
            long = 'высокая'    
    
        st.subheader(f'Опасность в краткосрочной перспективе: {short}')
    
        st.subheader(f'Опасность в долгосрочной перспективе: {long}')
