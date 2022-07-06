import streamlit as st
from streamlit_option_menu import option_menu
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
import pandas as pd
import numpy as np
import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from matplotlib.colors import ListedColormap
from sklearn import metrics
from matplotlib import colors
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from numpy import mean,std
from statistics import mode
from xgboost.sklearn import XGBClassifier

#########################################
st.set_page_config(layout="wide")
#Remove the Hamburger menu for better user interface
# st.markdown(""" <style>
# #MainMenu {visibility: hidden;}
# footer {visibility: hidden;}
# </style> """, unsafe_allow_html=True)

#Initiate the option menu that will be called in every section later on
rad = option_menu(
            menu_title=None,  # required
            options=["Home Page", "Dataset Info", "Exploratory Analysis", "Predictions"],
            icons=["house", "book","activity", "heart"],
            menu_icon="cast",
            default_index=0,
            orientation="horizontal",
        )
padding=0

#Create the upload data button and assign the data as a global component so it can be used in every section
uploaded_file=st.sidebar.file_uploader(label="Upload your Data", accept_multiple_files=False, type=['csv', 'xlsx'])
#Initiate a try function to assign the uploeaded file to the variable "data" and accomodate for csv and excel
global data
if uploaded_file is not None:
    uploaded_file.seek(0)
    print(uploaded_file)
    try:
        data=pd.read_csv(uploaded_file)

    except Exception as e:
        print(e)
        data=pd.read_excel(uploaded_file)

######################################### HOME PAGE
if rad=="Home Page":
    st.markdown(f""" <style>
        .reportview-container .main .block-container{{
            padding-top: {padding}rem;
            padding-right: {padding}rem;
            padding-left: {padding}rem;
            padding-bottom: {padding}rem;
        }} </style> """, unsafe_allow_html=True)
    image, title= st.columns((1,10))
    with image:
        st.image("https://i0.wp.com/gbsn.org/wp-content/uploads/2020/07/AUB-logo.png?ssl=1", width=200)
    with title:
        st.markdown("<h1 style='text-align: center; color: black;'>Heart Disease Causes and Prediction", unsafe_allow_html=True)
    col1,col2,col3= st.columns((3,5,2))
    with col1:
        st.write(' ')
    with col2:
        st.image("https://i.ytimg.com/vi/5aDqztmYpPw/maxresdefault.jpg", use_column_width=True)
    with col3:
        st.write(' ')
    st.markdown("Cardiovascular diseases (CVDs), principally ischemic heart disease (IHD) and stroke, are the leading cause of global mortality and a major contributor to disability. the number of CVD deaths steadily increased from 12.1 million in 1990 to 18.6 million in 2019. Disability-adjusted life years **(DALYs)** and years of life lost **(YLL)** from CVDs almost **doubled** from 1990 to 2019. Cardiovascular diseases remain the **leading cause** of disease burden in the world. Based on the presented facts, there is an **urgent** need to focus on implementing existing cost-effective policies and unconventional tools like the one presented in this app to reduce premature mortality.")

######################################### DATASET INFO
if rad=="Dataset Info":
    c1,c2= st.columns(2)
    try:
        with c1:
            """##### Dataset"""
            AgGrid(data, height=300, theme= "streamlit")
            rows= data.shape[0]
            columns= data.shape[1]
            st.write(f'The dataset has {rows} rows and {columns} columns')
            nan= data.isnull().sum().sum()
            st.write(f"The uploaded dataset has a total of {nan} missing values accross all columns!")
            diz= data["Target"].value_counts()
            st.write(f"The number of patients who suffer from heart disease is {diz.loc[0]} compared to {diz.loc[1]} who do not in the dataset.")
            
            
        with c2:
            """##### Major Features, Summary Statistics"""
            num_features= data[["Age", "Rest_Blood_Pressure", "Cholesterol", "Max_Heart_Rate"]]
            stats= num_features.describe()
            st.table(stats)

    except Exception as e:
        print(e)
        st.header("Please Upload a File in the Sidebar")
######################################### Explore Dataset (Outliers, distributions, relationships)
################################################################################    Exploratory Data Analysis ###################################################################
if rad=="Exploratory Analysis":
######################################### Density Plots
#Age Filter in sidebar

    with st.sidebar:
        data['Age'] = pd.cut(data['Age'], bins=[28,49,69,99], labels=['30-50', '50-70', '70+'])
        data["Age"].astype("category")
        ages_list=data["Age"].unique().tolist()
        ages = st.container()
        all = st.checkbox("Select all", value=True)

        if all:
            selected_options = ages.multiselect("Select Age Group",
                ages_list, ages_list)
        else:
            selected_options =  ages.multiselect("Select Age Group",
                ages_list)
        data = data.loc[data["Age"].isin(selected_options)]
    #Gender filter in sidebar
    with st.sidebar:
        gender_list=data["Gender"].unique().tolist()
        gender = st.container()
        all = st.checkbox("All Data", value=True)

        if all:
            selected_optionss = gender.multiselect("Select Gender",
                gender_list, gender_list)
        else:
            selected_optionss =  gender.multiselect("Select Gender",
                gender_list)
        data = data.loc[data["Gender"].isin(selected_optionss)]
    a1,a2,a3 =st.columns(3)
    with a1:

        """##### Density Plots by Gender"""

        # Scale the data using robust scaler to take into account outliers and render all data within same scale
        distribution=st.selectbox(" ", ["Resting Blood Pressure", "Cholesterol", "Max Heart Rate"])

        if distribution == "Resting Blood Pressure":
            #color_discrete_map = {'Male': 'rgb(101,28,50)', 'Female': 'rgb(9,40,30)'}
            hist1= px.histogram(data, x="Rest_Blood_Pressure", color="Gender",category_orders={'Gender':['Female', 'Male']})
            #hist1.update_traces(opacity=0.75)
            hist1.update_layout(legend=dict(yanchor="top", y=0.9, xanchor="left", x=0.65), autosize=False,
        width=500,
        height=400,
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=10,
            pad=0
        ))
            st.write(hist1, use_container_width=True)
            avg0= data[data['Gender'] == "Male"]['Rest_Blood_Pressure'].mean()
            avg1=data[data['Gender'] == "Female"]['Rest_Blood_Pressure'].mean()
            mode= mode(data["Rest_Blood_Pressure"])
            st.markdown(f"The average resting blood pressure level of Men is {avg0:.0f}, Women {avg1:.0f} and the mode observation is at {mode}")


        if distribution == "Cholesterol":
            #color_discrete_map = {'Male': 'rgb(31,119,180)', 'Female': 'rgb(214,39,40)'}
            hist1= px.histogram(data, x="Cholesterol", color="Gender",category_orders={'Gender':['Female', 'Male']})
            #hist1.update_traces(opacity=0.75)
            hist1.update_layout(legend=dict(yanchor="top", y=0.9, xanchor="left", x=0.75), autosize=False,
        width=500,
        height=400,
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=10,
            pad=0
        ))
            st.write(hist1, use_container_width=True)
            avg0= data[data['Gender'] == "Male"]['Cholesterol'].mean()
            avg1=data[data['Gender'] == "Female"]['Cholesterol'].mean()
            mode= mode(data["Cholesterol"])
            st.markdown(f"The average Cholesterol level of Men is {avg0:.0f}, Women {avg1:.0f} and the mode observation is at {mode}")


        if distribution == "Max Heart Rate":
            #color_discrete_map = {'Male': 'rgb(31,119,180)', 'Female': 'rgb(214,39,40)'}
            hist1= px.histogram(data, x="Max_Heart_Rate", color="Gender",category_orders={'Gender':['Female', 'Male']})
            #hist1.update_traces(opacity=0.75)
            hist1.update_layout(legend=dict(yanchor="top", y=0.9, xanchor="left", x=0.75), autosize=False,
        width=500,
        height=400,
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=10,
            pad=0
        ))
            st.write(hist1, use_container_width=True)

            avg0= data[data['Gender'] == "Male"]['Max_Heart_Rate'].mean()
            avg1=data[data['Gender'] == "Female"]['Max_Heart_Rate'].mean()
            mode= mode(data["Max_Heart_Rate"])
            st.markdown(f"The average maximum heart rate of Men is {avg0:.0f}, Women {avg1:.0f} and the mode observation is at {mode}")
######################################### CVD by Gender
        #Groupby the data using the target and gender to have more flexibility in building the plot
        """#####  CVDs by Gender"""
        b = data.groupby(by=["Gender", "Target"]).size().reset_index(name="Counts").sort_values(by='Counts', ascending=False)
        #change the type of target column to category to let plotly color it based on that and not numerical (no heatmap)
        b['Target']=b['Target'].astype("category")
        fig2=px.bar(b, x="Gender", y="Counts", color="Target",text_auto=True)
        fig2.update_layout(legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.7),autosize=False, width=420, height=380,margin=dict(
        l=0,
        r=0,
        b=0,
        t=10,
        pad=0
        ))
        #change the labels of legend
        newnames={"0": "No CVD", "1":"CVD" }
        fig2.for_each_trace(lambda t: t.update(name = newnames[t.name]))
        st.write(fig2, use_container_width=True)
        #explain results
        st.markdown("42% of Males and 72% of Females in the dataset suffer from Cardio Vascular Diseases")

######################################### Correlation Matrix
    with a2:
        data.drop(["Thal", "OldPeak"], axis=1, inplace=True)
        #Heatmap- Correlation Matrix to check the relationships between numerical features
        """##### Correlation"""
        # fig1=plt.figure(figsize=(10,8))
        corr = data.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        dt=go.Heatmap(
        z=corr.mask(mask),
        x=corr.columns,
        y=corr.columns,
        colorscale='Blues',
        zmin=-1,
        zmax=1
        )
        layout=go.Layout(width=420, height=495, yaxis_autorange='reversed',margin=dict(
                l=30    ,
                r=0,
                b=0,
                t=10,
                pad=0
            ))
        correlation=go.Figure(data=[dt], layout=layout)
        correlation.update_yaxes(visible=False)
        correlation.update_traces(colorbar = dict(orientation='h', y = -0.25, x = 0.5), showscale=False)

        st.write(correlation, use_container_width=True)

        def get_top_abs_correlations(df, n=5):
            au_corr = df.corr().abs().unstack()
            labels_to_drop = get_redundant_pairs(df)
            au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
            return au_corr[0:n]

        st.markdown(f"Most correlated features with CVD are chest pain type, max heart rate and the slope of the peak exercise ST segment")
######################################### CVD by # of colored Vessels during fluoroscopy
        #Groupby the data using the target and gender to have more flexibility in building the plot
        """#####  CVDs & Fluoroscopy"""
        c = data.groupby(by=["Major_Vessels_Colored", "Target"]).size().reset_index(name="Counts").sort_values(by='Counts', ascending=False)
        #change the type of target column to category to let plotly color it based on that and not numerical (no heatmap)
        c['Target']=c['Target'].astype("category")
        fig3=px.bar(c, x="Major_Vessels_Colored", y="Counts", color="Target",text_auto=True)
        fig3.update_layout(legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.7),autosize=False, width=420, height=380,margin=dict(
        l=0,
        r=0,
        b=0,
        t=10,
        pad=0
        ))
        #change the labels of legend
        newnames={"0": "No CVD", "1":"CVD"}
        fig3.for_each_trace(lambda t: t.update(name = newnames[t.name]))
        st.write(fig3, use_container_width=True)
        #explain results
        st.markdown("The lower the number of vessels colored during fluoroscopy, the higher the chance of CVD in patient")

    with a3:
        """#####  Age & CVDs"""
        #Segment age column to three main groups to act as a category
        #Select observations that are only attained by CVDs (of class 1) out of the created age groups above
        df=data.loc[data['Target'] == 1, 'Age']
        df2=df.value_counts() .reset_index()
        #Rename columns of value counts so that I can use them in plot
        df2.columns=["Age", "Counts"]
        fig = px.pie(df2, values="Counts", names="Age")
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.8), legend_title= "Age",autosize=False, width=380, height=500,margin=dict(
        l=60,
        r=0,
        b=0,
        t=0,
        pad=0
        ))
        st.write(fig, use_container_width=True)
        st.markdown("People aged 50 to 70 years are the most hit by CVDs")
        st.markdown("")

######################################### Angina & CVDs
        """#####  CVDs & Angina"""
        c = data.groupby(by=["Exercise_Angina", "Target"]).size().reset_index(name="Counts").sort_values(by='Counts', ascending=False)
        #change the type of target column to category to let plotly color it based on that and not numerical (no heatmap)
        c['Target']=c['Target'].astype("category")
        fig2=px.bar(c, x="Exercise_Angina", y="Counts", color="Target",text_auto=True)
        fig2.update_layout(legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.7),autosize=False, width=420, height=380,margin=dict(
        l=0,
        r=0,
        b=0,
        t=10,
        pad=0
        ))
        #change the labels of legend
        newnames={"0": "No CVD", "1":"CVD" }
        fig2.for_each_trace(lambda t: t.update(name = newnames[t.name]))
        st.write(fig2, use_container_width=True)
        #explain results
        st.markdown("Patients with no exercise enduced angina are more prone to CVD")

######################################### Age & CVD
# ################################################################################    Machine Learning ###################################################################

################################################################################   PREDICTIONS ###################################################################
if rad == "Predictions":
    st.markdown("""Please input below the details of your patient and hit "Predict" to use the power of Machine Learning to check if he/she suffers from heart disease""")
    #Convert the data type of certain features to their proper formats
    data['Target']=data['Target'].astype("category")
    data['Fasting_Blood_Sugar']=data['Fasting_Blood_Sugar'].astype("category")
    data['Gender']=data['Gender'].astype("category")
    data['Exercise_Angina']=data['Exercise_Angina'].astype("category")
    data['Chest_Pain_Type']=data['Chest_Pain_Type'].astype("category")
    data['Rest_ECG']=data['Rest_ECG'].astype("category")
    data['Thal']=data['Thal'].astype("category")
    data['Slope']=data['Slope'].astype("category")

    #Split the data between the target variable (y) and everything else (X)
    X=data.loc[:, data.columns != 'Target']
    y=data['Target']

    #Split X into categorical and numerical variables for better Preprocessing pipelines
    num_features = X.select_dtypes(include=['int64', 'int32', 'float64'])
    cat_features = X.select_dtypes(include=['object','category'])
    #Note that I will not remove outliers since patients with outlier results might be the ones that are at risk in CVDs
    numerical_transformer = Pipeline(steps=[
    ('num_imputer', SimpleImputer(missing_values = np.nan, strategy='mean')),
    ('normalizer', Normalizer())
    ])
    #Pipeline for categorical features preprocessing (imputing missing values by mode, encoding using OHE Dummies)
    categorical_transformer = Pipeline(steps=[
        ('cat_imputer', SimpleImputer(missing_values = np.nan, strategy='most_frequent')),
        ('encoder',OneHotEncoder(drop='first',handle_unknown='ignore',sparse=False))
        ])
    #Fitting the numerical and categorical features datasets into their corresponding transformers
    transformer = ColumnTransformer( transformers=[
            ('numerical', numerical_transformer, num_features.columns),
            ('categorical',categorical_transformer,cat_features.columns)]
            ,remainder='passthrough')
    #Define a function to bridge manual inputs from the app to actual data entries for the model's evaluation
    def input_features():
        gender= st.radio("Gender", ("Female", "Male"))
        age= st.number_input("Age", 1, 90)
        Chest_Pain_Type= st.radio("Chest Pain Severity (3 is highest) ", ("0", "1", "2", "3"))
        Rest_Blood_Pressure= st.number_input("Resting Blood Pressure", 60, 250)
        cholesterol= st.number_input("Cholesterol Level", 100, 600)
        fast= st.radio("Fasting Blood Sugar > 120 mg/dl? ", ("Yes", "No"))
        ecg= st.radio("ECG", ("0", "1", "2"))
        max_heart_rate= st.number_input("Max Heart Rate", 150, 250)
        exangina= st.radio("Exercise Induced Angina? ", ("Yes", "No"))
        oldpeak= st.number_input("Old Peak", 0, 10)
        slope= st.number_input("slope", 0, 2)
        vesselscolored= st.number_input("How many Major Vessels were Colored by Fluoroscopy? ", 0, 3)
        Thal= st.number_input("Thal", 0, 3)

        #Match the input data from app (seen above) to the names of the initial data in the form of dictionary
        data ={'Gender':gender,'Age':age,'Chest_Pain_Type':Chest_Pain_Type, 'Rest_Blood_Pressure':Rest_Blood_Pressure,'Cholesterol':cholesterol, 'Fasting_Blood_Sugar':fast, 'Rest_ECG':ecg, 'Max_Heart_Rate': max_heart_rate, 'Exercise_Angina': exangina, 'OldPeak': oldpeak, 'Slope':slope, 'Major_Vessels_Colored': vesselscolored, 'Thal': Thal}

        features=pd.DataFrame(data, index=[0])
        return features
    df1=input_features()

    #Initiate the predictive model: XGBoost since it was the best performer
    #model = LogisticRegression(max_iter=1000, random_state=1)
    model=XGBClassifier()
    ultimate_pipeline= Pipeline(steps=[('transformer', transformer), ('model', model)])
    cv= RepeatedStratifiedKFold(n_splits=10, n_repeats=3,  random_state =1)
    ultimate_pipeline.fit(X, y)
    pred=ultimate_pipeline.predict(df1)
    #Calculate the AUC of prediction
    auc = cross_val_score(ultimate_pipeline, X, y, scoring='roc_auc', cv=cv, n_jobs=-1).mean()   #I used repeated stratified method since the data size is not very significant
    if st.button("Predict"):
        st.success("Done !")
        st.subheader(f"The Patient {str(pred)} at risk of Cardiovascular Disease with an average ROC AUC of {auc*100:.2f}% ")
        st.markdown("**Legend: [0]= IS NOT, [1]= IS**")
