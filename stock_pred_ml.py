import streamlit as st
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fbprophet import Prophet
from fbprophet.diagnostics import performance_metrics
from fbprophet.diagnostics import cross_validation
from fbprophet.plot import plot_cross_validation_metric
import base64
#sktime models
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import smape_loss
from sktime.utils.plotting.forecasting import plot_ys
from sktime.forecasting.arima import AutoARIMA
import matplotlib
matplotlib.use("Agg")
import emoji
import random
import time
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDClassifier

from pmdarima import auto_arima 
 
from statsmodels.tools.eval_measures import rmse 
import numpy as np
import base64
#sktime models

from statsmodels.tsa.statespace.sarimax import SARIMAX 

import warnings
warnings.filterwarnings('ignore')
st.set_option('deprecation.showfileUploaderEncoding', False)

st.title('Time Series - ML')

@st.cache
def tsplot(y, lags=None, figsize=(20, 12), style='bmh'):
    """
        Plot time series, its ACF and PACF
        
        y - timeseries
        lags - how many lags to include in ACF, PACF calculation
    """
        
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        
        y.plot(ax=ts_ax)
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()



## Function for splitting the dataset        
def splitdata(X,y):
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
    return X_train, X_test, y_train, y_test

  
## Function for labelencoding
def encode(dataframe):
    
    for col in dataframe.columns:
        if dataframe[col].dtype.name == 'object':
            le = LabelEncoder()
            dataframe[col] = le.fit_transform(dataframe[col])
    return dataframe


## Function for Scaling
def scale(X_train,X_test):
    
    sc = StandardScaler()   
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test


## Function for oversampling
def oversample(X,y):
    
    smote = random.choice([SMOTE(),RandomOverSampler()])
    X,y = smote.fit_resample(X,y)
    return X,y

def null_values(df):
    null_test = (df.isnull().sum(axis = 0)/len(df)).sort_values(ascending=False).index
    null_data_test = pd.concat([
        df.isnull().sum(axis = 0),
        (df.isnull().sum(axis = 0)/len(df)).sort_values(ascending=False),
        df.loc[:, df.columns.isin(list(null_test))].dtypes], axis=1)
    null_data_test = null_data_test.rename(columns={0: '# null', 
                                        1: '% null', 
                                        2: 'type'}).sort_values(ascending=False, by = '% null')
    null_data_test = null_data_test[null_data_test["# null"]!=0]
    
    return null_data_test

def types(df):
    return pd.DataFrame(df.dtypes, columns=['Type'])

def forecasting_autoarima(y_train, y_test, s):
    fh = np.arange(len(y_test)) + 1
    forecaster = AutoARIMA(sp=s)
    forecaster.fit(y_train)
    y_pred = forecaster.predict(fh)
    plot_ys(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
    st.pyplot()

def main():
    st.sidebar.title("What to do")
    activities = ["Exploratory Data Analysis", "Plotting and Visualization", "Building Model", "About"]
    choice = st.sidebar.selectbox("Select Activity", activities)
    # Add a slider to the sidebar:
    st.sidebar.markdown("# Lang")
    x = st.sidebar.slider(
        'Select a lang for ACF and PACF analysis',
        50, 60
    )
    # Add a slider to the sidebar:
    st.sidebar.markdown("# Seasonal")
    s = st.sidebar.slider(
        'Select a seasonal parameter from previous ACF and PACF analysis',
        24, 48
    )
    # cloud logo
    st.sidebar.title("Built on:")
    
    # Upload file
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None and choice == "Exploratory Data Analysis":
        data = pd.read_csv(uploaded_file)
        st.subheader(choice)
        # Show dataset
        if st.checkbox("Show Dataset"):
            rows = st.number_input("Number of rows", 5, len(data))
            st.dataframe(data.head(rows))
        # Show columns
        if st.checkbox("Columns"):
            st.write(data.columns)
        # Data types
        if st.checkbox("Column types"):
            st.write(types(data))
        # Show Shape
        if st.checkbox("Shape of Dataset"):
            data_dim = st.radio("Show by", ("Rows", "Columns", "Shape"))
            if data_dim == "Columns":
                st.text("Number of Columns: ")
                st.write(data.shape[1])
            elif data_dim == "Rows":
                st.text("Number of Rows: ")
                st.write(data.shape[0])
            else:
                st.write(data.shape)
        # Check null values in dataset
        if st.checkbox("Check null values"):
            nvalues = null_values(data)
            st.write(nvalues)
        # Show Data summary
        if st.checkbox("fill null values with 1 day back data"):
            data.fillna(method="bfill",inplace=True)
            nvalues = null_values(data)
            st.write(nvalues)
        if st.checkbox("Show Data Summary"):
            st.text("Datatypes Summary")
            st.write(data.describe())
        # Plot time series, ACF and PACF
        if st.checkbox("Select column as time series"):
            columns = data.columns.tolist()
            selected = st.multiselect("Choose", columns)
            series = data[selected]
            if st.button('Plot Time Series, ACF and PACF'):
                tsplot(series, lags=x)
                st.pyplot()

    elif uploaded_file is not None and choice == "Plotting and Visualization":
        st.subheader(choice)
        data = pd.read_csv(uploaded_file)
        data.fillna(method="bfill",inplace=True)
        df = data.copy()
        all_columns = df.columns.tolist()
        type_of_plot = st.selectbox("Select Type of Plot", ["area", "line", "scatter", "pie", "bar", "correlation", "distribution"]) 
        
        if type_of_plot=="line":
            select_columns_to_plot = st.multiselect("Select columns to plot", all_columns)
            cust_data = df[select_columns_to_plot]
            st.line_chart(cust_data)
        
        elif type_of_plot=="area":
            select_columns_to_plot = st.multiselect("Select columns to plot", all_columns)
            cust_data = df[select_columns_to_plot]
            st.area_chart(cust_data)  
        
        elif type_of_plot=="bar":
            select_columns_to_plot = st.multiselect("Select columns to plot", all_columns)
            cust_data = df[select_columns_to_plot]
            st.bar_chart(cust_data)
        
        elif type_of_plot=="pie":
            select_columns_to_plot = st.selectbox("Select a column", all_columns)
            st.write(df[select_columns_to_plot].value_counts().plot.pie())
            st.pyplot()
        
        elif type_of_plot=="correlation":
            st.write(sns.heatmap(df.corr(), annot=True, linewidths=.5, annot_kws={"size": 7}))
            st.pyplot()

        elif type_of_plot=="scatter":
            st.write("Scatter Plot")
            scatter_x = st.selectbox("Select a column for X Axis", all_columns)
            scatter_y = st.selectbox("Select a column for Y Axis", all_columns)
            st.write(sns.scatterplot(x=scatter_x, y=scatter_y, data = df))
            st.pyplot()

        elif type_of_plot=="distribution":
            select_columns_to_plot = st.multiselect("Select columns to plot", all_columns)
            st.write(sns.distplot(df[select_columns_to_plot]))
            st.pyplot()

    elif uploaded_file is not None and choice == "Building Model":
        st.subheader(choice)
        data = pd.read_csv(uploaded_file)
        data.fillna(method="bfill",inplace=True)
        df = data.copy()
        st.write("Select the columns to use for training")
        columns = df.columns.tolist()
        selected_column = st.multiselect("Select Columns", columns)
        new_df = df[selected_column]
        st.write(new_df)
        
        if st.checkbox("Train/Test Split"):
            try:
                y_train, y_test = temporal_train_test_split(new_df.T.iloc[0])
                st.text("Train Shape")
                st.write(y_train.shape)
                st.text("Test Shape")
                st.write(y_test.shape)
                plot_ys(y_train, y_test, labels=["y_train", "y_test"])
                st.pyplot()
            except IndexError:
                st.write("First select timeseries column to train, for further operation")
        activities = ["Select one","CLASSIFICATION","REGRESSION","Timeseries"]	
        st.subheader("Select the type of model {}")
        choice = st.selectbox('',activities)
        
        
        if choice == 'CLASSIFICATION':
            classification_activities = ["Select one",'Random Forest Classifier','Decision Tree Classifier','SVC',
                           'SGD Classifier','Gradient Boosting Classifier',
                           'Adaboost Classifier']
            st.subheader("Select a Classification model to train your Dataset on {}".format(emoji.emojize(":eyes:")))
            classifier_choice = st.selectbox("",classification_activities)
            st.subheader("Select the hyper-parameter {}".format(emoji.emojize(":smiley:")))
            if classifier_choice == 'Random Forest Classifier':
                columns = data.columns.tolist()
                selected_column = st.multiselect("Select feature", columns)
                selected_column1 = st.multiselect("Select label", columns)
                new_df = data[selected_column]
                new_df1 = data[selected_column1]
                max_depth= st.slider(label='max_depth', min_value=0, max_value=16, step=1)
                n_estimators= st.slider(label='n_estimators', min_value=100, max_value=800, step=50)
                min_samples_split= st.slider(label='min_samples_split', min_value=0, max_value=16, step=1)
        
            elif classifier_choice == 'Decision Tree Classifier':
                columns = data.columns.tolist()
                selected_column = st.multiselect("Select feature", columns)
                selected_column1 = st.multiselect("Select label", columns)
                new_df = data[selected_column]
                new_df1 = data[selected_column1]
                max_depth= st.slider(label='max_depth', min_value=0, max_value=16, step=1)
                min_samples_split= st.slider(label='min_samples_split', min_value=0, max_value=16, step=1)
                criterion = st.selectbox('criterion',['gini','entropy'])
        
            elif classifier_choice == 'SVC':
                columns = data.columns.tolist()
                selected_column = st.multiselect("Select feature", columns)
                selected_column1 = st.multiselect("Select label", columns)
                new_df = data[selected_column]
                new_df1 = data[selected_column1]
                c= st.slider(label='C', min_value=0, max_value=16, step=1)
                degree= st.slider(label='degree', min_value=1, max_value=10, step=1)
                kernel = st.selectbox('kernel',['linear', 'poly', 'rbf', 'sigmoid'])

            elif classifier_choice == 'SGD Classifier':
                columns = data.columns.tolist()
                selected_column = st.multiselect("Select feature", columns)
                selected_column1 = st.multiselect("Select label", columns)
                new_df = data[selected_column]
                new_df1 = data[selected_column1]
                loss = st.selectbox('loss',['hinge', 'log', 'modified_huber', 'squared_hinge'])
                penalty = st.selectbox('penalty',['l1','l2','elasticnet'])
                alpha= st.text_input(label='alpha(enter a value between 0.0001 to 0.001)',value='0.0001')
                alpha = float(alpha)
            
            elif classifier_choice == 'Gradient Boosting Classifier':
                columns = data.columns.tolist()
                selected_column = st.multiselect("Select feature", columns)
                selected_column1 = st.multiselect("Select label", columns)
                new_df = data[selected_column]
                new_df1 = data[selected_column1]
                n_estimators= st.slider(label='n_estimators', min_value=100, max_value=800, step=50)
                learning_rate= st.slider(label='learning_rate', min_value=0.1, max_value=1.0, step=0.05)
                criterion= st.selectbox('criterion', ['friedman_mse', 'mse', 'mae'])
            
            elif classifier_choice == 'Adaboost Classifier':
                columns = data.columns.tolist()
                selected_column = st.multiselect("Select feature", columns)
                selected_column1 = st.multiselect("Select label", columns)
                new_df = data[selected_column]
                new_df1 = data[selected_column1]
                n_estimators= st.slider(label='n_estimators', min_value=50, max_value=800, step=50)
                learning_rate= st.slider(label='learning_rate', min_value=0.5, max_value=3.0, step=0.05)
            
            cross_validation = st.slider(label='cross validation (higher the number more the time taken to train)', min_value=1, max_value=10, step=1)
            submit = st.button('TRAIN')
        
            if submit:
                
                data = encode(data)
                
                
            
                X = data[selected_column]
                y = data[selected_column1]
            
                X,y = oversample(X,y)
            
                splitreturn = splitdata(X,y)
                X_train,X_test,y_train,y_test = splitreturn[0],splitreturn[1],splitreturn[2],splitreturn[3]
            
                scalereturn = scale(X_train,X_test)
                X_train,X_test = scalereturn[0],scalereturn[1]
            
                st.write("Give us some {} to build your project".format(emoji.emojize(":watch:")))
            
            ## Function for RandomForestClassifier
                def randomforestclassifier(X_train,X_test,y_train,y_test,max_depth=None,n_estimators=100,min_samples_split=2,cross_validation=2):
                    classifier = RandomForestClassifier()
                    clffit = classifier.fit(X_train,y_train)
                    parameters = [{'max_depth':[max_depth],'n_estimators':[n_estimators],'min_samples_split':[min_samples_split]}]
                    gs = GridSearchCV(estimator = clffit,
                                  param_grid = parameters,
                                  n_jobs = -1,
                                  scoring = 'accuracy',
                                  cv = cross_validation)
                    gs.fit(X_train, y_train)
                    classifier = gs.best_estimator_
                    c = classifier.fit(X_train,y_train)
                    return classifier.predict(X_test), gs.best_params_, c
                
            
            ## Function for DecisionTreeCLassifier
                def decisiontreeclassifier(X_train,X_test,y_train,y_test,max_depth=None,min_samples_split=2,criterion='gini',cross_validation=2):
                    classifier = DecisionTreeClassifier()
                    clffit = classifier.fit(X_train,y_train)
                    parameters = [{'max_depth':[max_depth],'min_samples_split':[min_samples_split],'criterion':[criterion]}]
                    gs = GridSearchCV(estimator = clffit,
                                      param_grid = parameters,
                                      n_jobs = -1,
                                      scoring = 'accuracy',
                                      cv = cross_validation)
                    gs.fit(X_train, y_train)
                    classifier = gs.best_estimator_
                    c = classifier.fit(X_train,y_train)
                    return classifier.predict(X_test), gs.best_params_, c
                
            
            ## Function for SVC
                def svc(X_train,X_test,y_train,y_test,c=1.0,degree=3,kernel='rbf',cross_validation=2):
                    
                    classifier = SVC()
                    clffit = classifier.fit(X_train,y_train)
                    parameters = [{'C':[c],'degree':[degree],'kernel':[kernel]}]
                    gs = GridSearchCV(estimator = clffit,
                                      param_grid = parameters,
                                      n_jobs = -1,
                                      scoring = 'accuracy',
                                      cv = cross_validation)
                    gs.fit(X_train, y_train)
                    classifier = gs.best_estimator_
                    classifier.fit(X_train,y_train)
                    return classifier.predict(X_test), gs.best_params_, classifier.fit(X_train,y_train)
                
            
            ## Function for sgdclassifier
                def sgdclassifier(X_train,X_test,y_train,y_test,loss='hinge',penalty='l2',alpha=0.0001,cross_validation=2):
                    
                    classifier = SGDClassifier()
                    clffit = classifier.fit(X_train,y_train)
                    parameters = [{'penalty':['l2'],'loss':[loss],'alpha':[alpha]}]
                    gs = GridSearchCV(estimator = clffit,
                                      param_grid = parameters,
                                      n_jobs = -1,
                                      scoring = 'accuracy',
                                      cv = cross_validation)
                    gs.fit(X_train, y_train)
                    classifier = gs.best_estimator_
                    c = classifier.fit(X_train,y_train)
                    return classifier.predict(X_test), gs.best_params_, c
                
                
                ## Function for gradientboostingclassifier
                def gradientboostingclassifier(X_train,X_test,y_train,y_test,n_estimators=100,learning_rate=0.1,criterion='friedman_mse',cross_validation=2):
                    
                    classifier = GradientBoostingClassifier()
                    clffit = classifier.fit(X_train,y_train)
                    parameters = [{'n_estimators':[n_estimators],'learning_rate':[learning_rate],'criterion':[criterion]}]
                    gs = GridSearchCV(estimator = clffit,
                                      param_grid = parameters,
                                      n_jobs = -1,
                                      scoring = 'accuracy',
                                      cv = cross_validation)
                    gs.fit(X_train, y_train)
                    classifier = gs.best_estimator_
                    c = classifier.fit(X_train,y_train)
                    return classifier.predict(X_test), gs.best_params_, c
                
                
                ## Function for adaboost
                def adaboostclassifier(X_train,X_test,y_train,y_test,n_estimators=50,learning_rate=1,cross_validation=2):
                    
                    classifier = AdaBoostClassifier()
                    clffit = classifier.fit(X_train,y_train)
                    parameters = [{'n_estimators':[n_estimators],'learning_rate':[learning_rate]}]
                    gs = GridSearchCV(estimator = clffit,
                                      param_grid = parameters,
                                      n_jobs = -1,
                                      scoring = 'accuracy',
                                      cv = cross_validation)
                    gs.fit(X_train, y_train)
                    classifier = gs.best_estimator_
                    c = classifier.fit(X_train,y_train)
                    return classifier.predict(X_test), gs.best_params_, c
    
    
            ## Model functions
            
                if classifier_choice == 'Random Forest Classifier':
                    # max_depth= st.slider(label='max_depth', min_value=0.0, max_value=16.0, step=0.5)
                    # n_estimators= st.slider(label='n_estimators', min_value=0.0, max_value=16.0, step=0.5)
                    # min_samples_split= st.slider(label='min_samples_split', min_value=0.0, max_value=16.0, step=0.5)
                    classifier_output = randomforestclassifier(X_train,X_test,y_train,y_test,max_depth,n_estimators,min_samples_split)
                    
                elif classifier_choice == 'Decision Tree Classifier':
                    classifier_output = decisiontreeclassifier(X_train,X_test,y_train,y_test,max_depth,min_samples_split,criterion)
                
                elif classifier_choice == 'SVC':
                    classifier_output = svc(X_train,X_test,y_train,y_test,c,degree,kernel)
                
                elif classifier_choice == 'SGD Classifier':
                    classifier_output = sgdclassifier(X_train,X_test,y_train,y_test,loss,penalty,alpha)
                
                elif classifier_choice == 'Gradient Boosting Classifier':
                    classifier_output = gradientboostingclassifier(X_train,X_test,y_train,y_test,n_estimators,learning_rate,criterion)
                
                elif classifier_choice == 'Adaboost Classifier':
                    classifier_output = adaboostclassifier(X_train,X_test,y_train,y_test)
                    
                    
                 ### Time for printingout the result
                
                st.write('My system caught on {} training your model to get the output for you {}'.format(emoji.emojize(':fire:'), emoji.emojize(':satisfied:')))
                time.sleep(1.5)
                st.write('Be safe, wear a mask{}'.format(emoji.emojize(':mask:')))
                time.sleep(1.5)
                st.write('Your scores are here {}'.format(emoji.emojize(':raised_hands:')))
                time.sleep(1.5)
                st.write("\n")
                st.success('Accuracy score of {} is: {}'.format(classifier_choice,accuracy_score(y_test,classifier_output[0])))
                st.success('f1 score of {} is: {}'.format(classifier_choice,f1_score(y_test,classifier_output[0],average='weighted')))
                st.success('Recall score of {} is: {}'.format(classifier_choice,recall_score(y_test,classifier_output[0],average='weighted')))
                st.success('Precision score of {} is: {}'.format(classifier_choice,precision_score(y_test,classifier_output[0],average='weighted')))
                st.write('Selected parameters are: ',classifier_output[1])
                
                st.subheader("Code (adjust hyperparameters manually)")
                file = open('codes_to_display/'+classifier_choice+' Code.txt','r')
                classifier_code = file.read()
                st.code(classifier_code, language='python')
                file.close()
                
                st.subheader("Report")
                know = open('knowledge_to_display/'+classifier_choice+' Report.txt','rb')
                classifier_report = know.read().decode(errors='replace')
                st.code(classifier_report)
                know.close()
        if choice =='REGRESSION':
            regression_activities = ["Select one",'Linear Regressor','Ridge Regressor','Lasso Regressor',
                   'DecisionTree Regressor','Gradient Boosting Regressor']
            st.subheader("Select a Regression model to train your Dataset on {}".format(emoji.emojize(":eyes:")))
            regressor_choice = st.selectbox("",regression_activities)
            st.subheader("Select the hyper-parameter {}".format(emoji.emojize(":smiley:")))
            
            if regressor_choice=='Linear Regressor':
                columns = data.columns.tolist()
                selected_column = st.multiselect("Select feature", columns)
                selected_column1 = st.multiselect("Select label", columns)
                new_df = data[selected_column]
                new_df1 = data[selected_column1]
                normalize=st.selectbox('normalize',['True','False'])
                
            elif regressor_choice=='Ridge Regressor':
                columns = data.columns.tolist()
                selected_column = st.multiselect("Select feature", columns)
                selected_column1 = st.multiselect("Select label", columns)
                new_df = data[selected_column]
                new_df1 = data[selected_column1]
                max_iter=st.slider(label='max_iter',min_value=100,max_value=1500,step=100)
                solver=st.selectbox('solver',['auto','svd','cholesky','lsqr','sparse_cg','sag','saga'])
                alpha= st.text_input(label='alpha(enter a value between 0.01 to 10.0)',value='1')
                alpha = float(alpha)
                
            elif regressor_choice=='Lasso Regressor':
                columns = data.columns.tolist()
                selected_column = st.multiselect("Select feature", columns)
                selected_column1 = st.multiselect("Select label", columns)
                new_df = data[selected_column]
                new_df1 = data[selected_column1]
                max_iter=st.slider(label='max_iter',min_value=100,max_value=1500,step=100)
                selection=st.selectbox('selection',['cyclic','random'])
                alpha= st.text_input(label='alpha(enter a value between 0.01 to 10.0)',value='1')
                alpha=float(alpha)
                
            elif regressor_choice=='DecisionTree Regressor':
                columns = data.columns.tolist()
                selected_column = st.multiselect("Select feature", columns)
                selected_column1 = st.multiselect("Select label", columns)
                new_df = data[selected_column]
                new_df1 = data[selected_column1]
                min_samples_split= st.slider(label='min_samples_split', min_value=0, max_value=16, step=1)
                max_depth= st.slider(label='max_depth', min_value=0, max_value=16, step=1)
                criterion=st.selectbox('criterion',['mse','friedman_mse','mae'])
                
            elif regressor_choice=='Gradient Boosting Regressor':
                columns = data.columns.tolist()
                selected_column = st.multiselect("Select feature", columns)
                selected_column1 = st.multiselect("Select label", columns)
                new_df = data[selected_column]
                new_df1 = data[selected_column1]
                loss=st.selectbox('loss',['ls','lad','huber','quantile'])
                n_estimators=st.slider(label='n_estimators',min_value=100,max_value=800,step=50)
                learning_rate= st.text_input(label='learning_rate(enter a value between 0.001 to 1.0)',value='0.1')
                learning_rate = float(learning_rate)
    
            cross_validation = st.slider(label='cross validation (higher the number more the time taken to train)', min_value=1, max_value=10, step=1)
            submit = st.button('TRAIN')
            
            if submit:
                
                
            
                data = encode(data)
                
                
            
                X = data[selected_column]
                y = data[selected_column1]
                
                splitreturn = splitdata(X,y)
                X_train,X_test,y_train,y_test = splitreturn[0],splitreturn[1],splitreturn[2],splitreturn[3]
                
                scalereturn = scale(X_train,X_test)
                X_train,X_test = scalereturn[0],scalereturn[1]
                
                st.write("Give us some {} to build your project".format(emoji.emojize(":watch:")))
                
                #Linear Regression
                def linearregressor(X_train,X_test,y_train,y_test,normalize=False,cross_validation=2):
                    regressor=LinearRegression()
                    parameters=[{'normalize':[normalize]}]
                    regressor=GridSearchCV(regressor,parameters,scoring='r2',cv=cross_validation)
                    regressor.fit(X_train,y_train)
                    
                    return regressor.predict(X_test), regressor.best_params_, regressor.score(X_test,y_test)
                
                #Ridge Regression
                def ridgeregressor(X_train,X_test,y_train,y_test,max_iter=100,solver='auto',alpha=1.0,cross_validation=2):
                    regressor=Ridge()
                    parameters=[{'max_iter':[max_iter],'solver':[solver],'alpha':[alpha]}]
                    regressor=GridSearchCV(regressor,parameters,scoring='r2',cv=cross_validation)
                    regressor.fit(X_train,y_train)
                    
                    return regressor.predict(X_test), regressor.best_params_, regressor.score(X_test,y_test)
                    
                #Lasso Regression
                def lassoregressor(X_train,X_test,y_train,y_test,max_iter=100,selection='cyclic',alpha=1.0,cross_validation=2):
                    regressor=Lasso()
                    parameters=[{'max_iter':[max_iter],'selection':[selection],'alpha':[alpha]}]
                    regressor=GridSearchCV(regressor,parameters,scoring='r2',cv=cross_validation)
                    regressor.fit(X_train,y_train)
                    
                    return regressor.predict(X_test), regressor.best_params_, regressor.score(X_test,y_test)
                
                #Decision Tree Regression
                def decisiontreeregressor(X_train,X_test,y_train,y_test,min_samples_split=2,max_depth=6,criterion='friedman_mse',cross_validation=2):
                    regressor = DecisionTreeRegressor()
                    parameters=[{'max_depth':[max_depth],'min_samples_split':[min_samples_split],'criterion':[criterion]}]
                    regressor=GridSearchCV(regressor,parameters,scoring='r2',cv=cross_validation)
                    regressor.fit(X,y)
    
                    return regressor.predict(X_test), regressor.best_params_, regressor.score(X_test,y_test)
                
                #Gradient Boosting Regression
                def gradientboostingregressor(X_train,X_test,y_train,y_test,loss='ls',n_estimators=100,learning_rate=0.1,cross_validation=2):
                    regressor = GradientBoostingRegressor()
                    parameters ={'n_estimators':[n_estimators],'loss':[loss],'learning_rate':[learning_rate]}
                    regressor = GridSearchCV(regressor,parameters,scoring='r2', cv=cross_validation)
                    regressor.fit(X_train,y_train)
    
                    return regressor.predict(X_test), regressor.best_params_, regressor.score(X_test,y_test)
    
               
                ## Model functions
                
                if regressor_choice == 'Linear Regressor':
                    regressor_output = linearregressor(X_train,X_test,y_train,y_test,normalize)
                    
                elif regressor_choice == 'Ridge Regressor':
                    regressor_output = ridgeregressor(X_train,X_test,y_train,y_test,max_iter,solver,alpha)
                
                elif regressor_choice == 'Lasso Regressor':
                    regressor_output = lassoregressor(X_train,X_test,y_train,y_test,max_iter,selection,alpha)
                
                elif regressor_choice == 'DecisionTree Regressor':
                    regressor_output = decisiontreeregressor(X_train,X_test,y_train,y_test,min_samples_split,max_depth,criterion)
                
                elif regressor_choice == 'Gradient Boosting Regressor':
                    regressor_output = gradientboostingregressor(X_train,X_test,y_train,y_test,loss,n_estimators,learning_rate)
    
                 ### Time for printing out the result
                    
                st.write('My system caught on {} training your model to get the output for you {}'.format(emoji.emojize(':fire:'), emoji.emojize(':satisfied:')))
                time.sleep(1.5)
                st.write('Be safe, wear a mask{}'.format(emoji.emojize(':mask:')))
                time.sleep(1.5)
                st.write('Your scores are here {}'.format(emoji.emojize(':raised_hands:')))
                time.sleep(1.5)
                st.write("\n")
                st.success("r2/variance for {} is: {}".format(regressor_choice, regressor_output[2]))
                st.success("Mean Squared Error for {} is: {}".format(regressor_choice, mean_squared_error(y_test, regressor_output[0])))
                st.write('Selected parameters are: ',regressor_output[1])
                
                st.subheader("Code (adjust hyperparameters manually)")
                file = open('codes_to_display/'+regressor_choice+' Code.txt','r')
                regressor_code = file.read()
                st.code(regressor_code, language='python')
                file.close()
                
                st.subheader("Report")
                know = open('knowledge_to_display/'+regressor_choice+' Report.txt','rb')
                regressor_report = know.read().decode(errors='replace')
                st.code(regressor_report)
                know.close()
        if choice =='Timeseries':
                
                model_selection = st.selectbox("Model to train", [" ","AutoArima", "LSTM", "MLP", "RNN","fbprophet"])
                if model_selection == "AutoArima":
                    y_train, y_test = temporal_train_test_split(new_df.T.iloc[0])
                    forecasting_autoarima(y_train, y_test, s)
                elif model_selection == "fbprophet":
                    columns = df.columns.tolist()
                    select_columns_to_plot = st.multiselect("Select columns to plot", columns)
                    cust_data = df[select_columns_to_plot]
                    try:
                        cust_data.columns=['ds','y']
                        cust_data['ds'] = pd.to_datetime(cust_data['ds'],errors='coerce')
                        max_date = cust_data['ds'].max()
                        periods_input = st.number_input('How many periods would you like to forecast into the future?',
                        min_value = 1, max_value = 365)
                        m = Prophet()
                        m.fit(cust_data)
                        future = m.make_future_dataframe(periods=periods_input)
                        forecast = m.predict(future)
                        fcst = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                        fcst_filtered =  fcst[fcst['ds'] > max_date]    
                        st.write(fcst_filtered)
                        fig1 = m.plot(forecast)
                        st.write(fig1)
                        fig2 = m.plot_components(forecast)
                        st.write(fig2)
                    except ValueError:
                      st.write("Choose date and column to plot for further prediction")

    elif choice == "About":
        st.title("About")
        st.write("The app developed by Akash")
        st.write("Stack: Python, Streamlit, Docker, Kubernetes")

if __name__ == "__main__":
    main()