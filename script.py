import pandas as pd
df=pd.read_excel("C:/Users/DINESH/Desktop/Data for DS/vscode/car/onlinemlflow/ml_dl.xlsx")

y=df["price"]
x=df.drop("price",axis=1)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

models=[
    ("Gradient Boosting Regressor",
     {"n_estimators":150,"learning_rate":0.1},
     GradientBoostingRegressor(),
     (x_train, y_train),
     (x_test, y_test)),
    ("Random Forest Regressor",
     {"n_estimators":150,"max_depth":5},
     RandomForestRegressor(),
     (x_train, y_train),
     (x_test, y_test)),
    ("Support Vector Regressor",
     {"kernel":"rbf","C":1.0,"epsilon":0.1},
     SVR(),
     (x_train, y_train),
     (x_test, y_test)),
    ("XGBRegressor",
     {"n_estimators":100,"learning_rate":0.1},
     XGBRegressor(),
     (x_train, y_train),
     (x_test, y_test))
]

reports=[]
for model_name,params,model,train_set,test_set in models:
    x_train1=train_set[0]
    y_train1=train_set[1]
    x_test1=test_set[0]
    y_test1=test_set[1]
    model.set_params(**params)
    model.fit(x_train1,y_train1)
    y_pred=model.predict(x_test1)

    mse=mean_squared_error(y_test1,y_pred)
    mae=mean_absolute_error(y_test1,y_pred)
    r2=r2_score(y_test1,y_pred)
    reports.append((model_name,mse,mae,r2))


import mlflow
mlflow.set_experiment("Car_experiment_models")
mlflow.set_tracking_uri("http://127.0.0.1:5001")
for i,element in enumerate(models):
    model_name=element[0]
    params=element[1]
    model=element[2]
    report=reports[i]

    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        model,mse, mae, r2 = report
        mlflow.log_metrics({"MSE":mse,"MAE":mae,"R2":r2})
        if model_name=="XGBRegressor":
            mlflow.xgboost.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model,"model")
