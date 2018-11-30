import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report

data_url = r"C:\Users\dhlee\Desktop\Data Science Competition 2018\TEST\wine_quality\winequality-white.csv"

dataset = pd.read_csv(data_url)
df = pd.read_csv(data_url)
km_df = pd.read_csv(data_url)

dataset = dataset.drop_duplicates()
df = df.drop_duplicates()
km_df = df.drop_duplicates()

df["citric_rate"] = df["citric acid"]/(df["fixed acidity"] + df["volatile acidity"]) * 100

df.loc[(df["chlorides"] > df["chlorides"].mean()) & ((df["alcohol"] >= 11.5) | (df["residual sugar"] >= df["residual sugar"].mean())),"albarino"] = 1
df.loc[(df["alcohol"] < df["alcohol"].mean()) &(df["residual sugar"] < df["residual sugar"].mean()),"loureiro"] =1
df.loc[(df["density"] > df["density"].mean()) & (df["citric_rate"]>df["citric_rate"].mean()), "arinto"] = 1

df.loc[df["albarino"].isnull(),"albarino"] = 0
df.loc[df["loureiro"].isnull(),"loureiro"] = 0
df.loc[df["arinto"].isnull(),"arinto"] = 0

df.loc[df["residual sugar"] > 12 ,"albarino"] = 1
df.head(1)

len(df.loc[(df["albarino"] == 0) & (df["loureiro"] == 0) & (df["arinto"] == 0)])/len(df) * 100
len(df.loc[(df["albarino"] == 1) & (df["loureiro"] == 0) & (df["arinto"] == 0)])/len(df) * 100
len(df.loc[(df["albarino"] == 0) & (df["loureiro"] == 1) & (df["arinto"] == 0)])/len(df) * 100
len(df.loc[(df["albarino"] == 0) & (df["loureiro"] == 0) & (df["arinto"] == 1)])/len(df) * 100
len(df.loc[(df["albarino"] == 1) & (df["loureiro"] == 0) & (df["arinto"] == 1)])/len(df) * 100
len(df.loc[(df["albarino"] == 0) & (df["loureiro"] == 1) & (df["arinto"] == 1)])/len(df) * 100

df.loc[(df["albarino"] == 0) & (df["loureiro"] == 0) & (df["arinto"] == 0), "wine_type"] = "port_blend"
df.loc[(df["albarino"] == 1) & (df["loureiro"] == 0) & (df["arinto"] == 0), "wine_type"] = "albarino"
df.loc[(df["albarino"] == 0) & (df["loureiro"] == 1) & (df["arinto"] == 0), "wine_type"] = "loureiro"
df.loc[(df["albarino"] == 0) & (df["loureiro"] == 0) & (df["arinto"] == 1), "wine_type"] = "arinto-loureiro"
df.loc[(df["albarino"] == 1) & (df["loureiro"] == 0) & (df["arinto"] == 1), "wine_type"] = "albarino"
df.loc[(df["albarino"] == 0) & (df["loureiro"] == 1) & (df["arinto"] == 1), "wine_type"] = "arinto-loureiro"

df.loc[(df["wine_type"] == "port_blend") & (( df["density"] > df["density"].mean() ) | (df["citric_rate"]>df["citric_rate"].mean() ) ) &
((df["alcohol"] < df["alcohol"].mean()) | (df["residual sugar"] < df["residual sugar"].mean()) ) &
((df["chlorides"] < df["chlorides"].mean()) & ((df["alcohol"] < 11.5) & (df["residual sugar"] < df["residual sugar"].mean())) ), "wine_type"] = "arinto-loureiro"

df.groupby("wine_type")["wine_type"].count() / len(df) * 100
del df["albarino"]
del df["loureiro"]
del df["arinto"]

df["wine_type"] = df["wine_type"].replace({"port_blend":0,"albarino":1,"loureiro":2, "al_lu":3})

df = pd.merge(df, pd.get_dummies(df["wine_type"], prefix="wine_type"),left_index=True, right_index=True)
del df["wine_type"]
del df["citric_rate"]

df
#####################################################################################################################################
print("uniq_quality:",dataset['quality'].unique())
dataset.info()

sns.countplot(dataset['quality'])

#######################################################################################################################################

def draw_multivarient_plot(dataset, rows, cols, plot_type):
    column_names=dataset.columns.values
    number_of_column=len(column_names)
    fig, axarr=plt.subplots(rows,cols, figsize=(18,10))

    counter=0
    for i in range(rows):
        for j in range(cols):
            if 'violin' in plot_type:
                sns.violinplot(x='quality', y=column_names[counter],data=dataset, ax=axarr[i][j])
            elif 'box'in plot_type :
                sns.boxplot(x='quality', y=column_names[counter],data=dataset, ax=axarr[i][j])
            elif 'point' in plot_type:
                sns.pointplot(x='quality',y=column_names[counter],data=dataset, ax=axarr[i][j])
            elif 'bar' in plot_type:
                sns.barplot(x='quality',y=column_names[counter],data=dataset, ax=axarr[i][j])

            counter+=1
            if counter==(number_of_column-1,):
                break

draw_multivarient_plot(dataset,6,2,"box")
draw_multivarient_plot(dataset,6,2,"violin")
draw_multivarient_plot(dataset,6,2,"pointplot")
draw_multivarient_plot(dataset,6,2,"bar")
#######################################################################################################################################
def get_models():
    models=[]
    models.append(("LR",LogisticRegression()))
    # models.append(("NB",GaussianNB()))
    # models.append(("KNN",KNeighborsClassifier()))
    # models.append(("DT",DecisionTreeClassifier()))
    # models.append(("SVM rbf",SVC()))
    # models.append(("SVM linear",SVC(kernel='linear')))

    return models

def cross_validation_scores_for_various_ml_models(X_cv, y_cv):
    print("교차검증 성공률".upper())
    models=get_models()


    results=[]
    names= []

    for name, model in models:
        kfold=KFold(n_splits=10,random_state=22)
        cv_result=cross_val_score(model,X_cv, y_cv, cv=kfold,scoring="accuracy")
        names.append(name)
        results.append(cv_result)
        print("{} 모델의 교차검증, 정확도:{:0.2f}".format(name, cv_result.mean()))
        return cv_result.mean()

#######################################################################################################################################

dataset_temp=dataset.copy(deep=True)
X=dataset.drop('quality', axis=1)
y=dataset['quality']

X=StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=42)


cross_validation_scores_for_various_ml_models(X, y)

def SVM_GridSearch(X_train, X_test, y_train, y_test):
    best_score=0
    gammas=[0.001, 0.01, 0.1, 1, 10, 100]
    Cs=[0.001, 0.01, 0.1, 1, 10, 100]

    for gamma in gammas:
        for C in Cs:
            svm=SVC(kernel='rbf',gamma=gamma, C=C)
            svm.fit(X_train, y_train)


            score=svm.score(X_test, y_test)

            if score>best_score:
                y_pred=svm.predict(X_test)
                best_score=score
                best_params={'C':100, 'gamma':gamma}

    print("best score:",best_score)
    print("best params:",best_params)
    print("classification reports:\n",classification_report(y_test, y_pred))
#############################################################################################################################################

SVM_GridSearch(X_train, X_test, y_train, y_test)

#############################################################################################################################################


df_temp=df.copy(deep=True)
X=df.drop('quality', axis=1)
y=df['quality']

X=StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=42)

cross_validation_scores_for_various_ml_models(X, y)
# SVM_GridSearch(X_train, X_test, y_train, y_test)
#######################################################################################################################################
albarino = df.loc[df["wine_type_1"] == 1]

alb_temp=albarino.copy(deep=True)
X=albarino.drop('quality', axis=1)
y=albarino['quality']

X=StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=42)

cross_validation_scores_for_various_ml_models(X, y)

# SVM_GridSearch(X_train, X_test, y_train, y_test)

#######################################################################################################################################
loureiro = df.loc[df["wine_type_2"] == 1]

lou_temp=loureiro.copy(deep=True)
X=loureiro.drop('quality', axis=1)
y=loureiro['quality']

X=StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=42)

cross_validation_scores_for_various_ml_models(X, y)

# SVM_GridSearch(X_train, X_test, y_train, y_test)

#######################################################################################################################################
port_blend = df.loc[df["wine_type_0"] == 1]

port_temp=port_blend.copy(deep=True)
X=port_blend.drop('quality', axis=1)
y=port_blend['quality']

X=StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=42)

cross_validation_scores_for_various_ml_models(X, y)

# SVM_GridSearch(X_train, X_test, y_train, y_test)

#######################################################################################################################################

alb_lou = df.loc[df["wine_type_3"] == 0]

alb_lou_temp=alb_lou.copy(deep=True)
X=alb_lou.drop('quality', axis=1)
y=alb_lou['quality']

X=StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=42)

cross_validation_scores_for_various_ml_models(X, y)

# SVM_GridSearch(X_train, X_test, y_train, y_test)

#######################################################################################################################################

# from sklearn.cluster import KMeans
# km = KMeans(n_clusters = 10, init='random', n_init=1, verbose=1)
# km.fit(km_df)
# resultLable = km.labels_
# km_df['label'] = pd.DataFrame(resultLable)
# km_df['label'].isnull().sum()
#
# kmdf = km_df.dropna(axis = 0)
# kmdf["label"].unique()
# kmdf = pd.merge(kmdf, pd.get_dummies(df["label"], prefix="label"),left_index=True, right_index=True)
# del kmdf["label"]
#
# kmdf_temp=kmdf.copy(deep=True)
# X=kmdf.drop('quality', axis=1)
# y=kmdf['quality']
#
# X=StandardScaler().fit_transform(X)
#
# X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=42)
#
#
# cross_validation_scores_for_various_ml_models(X, y)
# SVM_GridSearch(X_train, X_test, y_train, y_test)

#######################################################################################################################################
dataset["quality"].unique()
dataset.loc[(dataset['quality']==3),'quality']=1

dataset.loc[(dataset['quality']==4),'quality']=2
dataset.loc[(dataset['quality']==5),'quality']=2
dataset.loc[(dataset['quality']==6),'quality']=2

dataset.loc[(dataset['quality']==7),'quality']=3
dataset.loc[(dataset['quality']==8),'quality']=3
dataset.loc[(dataset['quality']==9),'quality']=3

dataset["quality"].unique()

X=dataset.drop('quality', axis=1)
y=dataset['quality']

dataset_temp

dataset["quality"].unique()
X_temp=dataset.drop('quality', axis=1)
y_temp=dataset['quality']
X=StandardScaler().fit_transform(X)

X_train_temp, X_test_temp, y_train_temp, y_test_temp=train_test_split(X_temp, y_temp, test_size=0.3, random_state=42)

print("기본 데이터만 이용")
cross_validation_scores_for_various_ml_models(X_temp, y_temp)
# SVM_GridSearch(X_train_temp, X_test_temp, y_train_temp, y_test_temp)

#######################################################################################################################################

df.loc[(df['quality']==3),'quality']=1

df.loc[(df['quality']==4),'quality']=2
df.loc[(df['quality']==5),'quality']=2
df.loc[(df['quality']==6),'quality']=2

df.loc[(df['quality']==7),'quality']=3
df.loc[(df['quality']==8),'quality']=3
df.loc[(df['quality']==9),'quality']=3


X=df.drop('quality', axis=1)
y=df['quality']

df["quality"].unique()
X_temp=df.drop('quality', axis=1)
y_temp=df['quality']
X=StandardScaler().fit_transform(X)

X_train_temp, X_test_temp, y_train_temp, y_test_temp=train_test_split(X_temp, y_temp, test_size=0.3, random_state=42)

print("와인 분류 데이터 이용")
cross_validation_scores_for_various_ml_models(X_temp, y_temp)
# SVM_GridSearch(X_train_temp, X_test_temp, y_train_temp, y_test_temp)

#######################################################################################################################################
#######################################################################################################################################

albarino.loc[(albarino['quality']==3),'quality']=1

albarino.loc[(albarino['quality']==4),'quality']=2
albarino.loc[(albarino['quality']==5),'quality']=2
albarino.loc[(albarino['quality']==6),'quality']=2

albarino.loc[(albarino['quality']==7),'quality']=3
albarino.loc[(albarino['quality']==8),'quality']=3
albarino.loc[(albarino['quality']==9),'quality']=3

X=albarino.drop('quality', axis=1)
y=albarino['quality']

albarino["quality"].unique()
X_temp=albarino.drop('quality', axis=1)
y_temp=albarino['quality']
X=StandardScaler().fit_transform(X)

X_train_temp, X_test_temp, y_train_temp, y_test_temp=train_test_split(X_temp, y_temp, test_size=0.3, random_state=42)

print("와인 분류 데이터 이용 - albarino")
cross_validation_scores_for_various_ml_models(X_temp, y_temp)
# SVM_GridSearch(X_train_temp, X_test_temp, y_train_temp, y_test_temp)

#######################################################################################################################################

loureiro.loc[(loureiro['quality']==3),'quality']=1

loureiro.loc[(loureiro['quality']==4),'quality']=2
loureiro.loc[(loureiro['quality']==5),'quality']=2
loureiro.loc[(loureiro['quality']==6),'quality']=2

loureiro.loc[(loureiro['quality']==7),'quality']=3
loureiro.loc[(loureiro['quality']==8),'quality']=3
loureiro.loc[(loureiro['quality']==9),'quality']=3

X=loureiro.drop('quality', axis=1)
y=loureiro['quality']

lou_temp["quality"].unique()
X_temp=lou_temp.drop('quality', axis=1)
y_temp=lou_temp['quality']
X=StandardScaler().fit_transform(X)

X_train_temp, X_test_temp, y_train_temp, y_test_temp=train_test_split(X_temp, y_temp, test_size=0.3, random_state=42)

print("와인 분류 데이터 이용 - loureiro")
cross_validation_scores_for_various_ml_models(X_temp, y_temp)
SVM_GridSearch(X_train_temp, X_test_temp, y_train_temp, y_test_temp)

#######################################################################################################################################

port_blend.loc[(port_blend['quality']==3),'quality']=1

port_blend.loc[(port_blend['quality']==4),'quality']=2
port_blend.loc[(port_blend['quality']==5),'quality']=2
port_blend.loc[(port_blend['quality']==6),'quality']=2

port_blend.loc[(port_blend['quality']==7),'quality']=3
port_blend.loc[(port_blend['quality']==8),'quality']=3
port_blend.loc[(port_blend['quality']==9),'quality']=3

X=port_temp.drop('quality', axis=1)
y=port_temp['quality']

port_temp["quality"].unique()
X_temp=port_temp.drop('quality', axis=1)
y_temp=port_temp['quality']
X=StandardScaler().fit_transform(X)

X_train_temp, X_test_temp, y_train_temp, y_test_temp=train_test_split(X_temp, y_temp, test_size=0.3, random_state=42)

print("와인 분류 데이터 이용 - portugal_white_blend")
cross_validation_scores_for_various_ml_models(X_temp, y_temp)
SVM_GridSearch(X_train_temp, X_test_temp, y_train_temp, y_test_temp)

#######################################################################################################################################

alb_lou.loc[(alb_lou['quality']==3),'quality']=1

alb_lou.loc[(alb_lou['quality']==4),'quality']=2
alb_lou.loc[(alb_lou['quality']==5),'quality']=2
alb_lou.loc[(alb_lou['quality']==6),'quality']=2

alb_lou.loc[(alb_lou['quality']==7),'quality']=3
alb_lou.loc[(alb_lou['quality']==8),'quality']=3
alb_lou.loc[(alb_lou['quality']==9),'quality']=3

X=alb_lou_temp.drop('quality', axis=1)
y=alb_lou_temp['quality']

alb_lou_temp["quality"].unique()
X_temp=alb_lou_temp.drop('quality', axis=1)
y_temp=alb_lou_temp['quality']
X=StandardScaler().fit_transform(X)

X_train_temp, X_test_temp, y_train_temp, y_test_temp=train_test_split(X_temp, y_temp, test_size=0.3, random_state=42)

print("와인 분류 데이터 이용 - albarino + loureiro")
cross_validation_scores_for_various_ml_models(X_temp, y_temp)
SVM_GridSearch(X_train_temp, X_test_temp, y_train_temp, y_test_temp)



draw_multivarient_plot(dataset,3,4,"bar")
draw_multivarient_plot(albarino,3,4,"bar")
draw_multivarient_plot(loureiro,3,4,"bar")
draw_multivarient_plot(port_blend,3,4,"bar")
draw_multivarient_plot(alb_lou,3,4,"bar")

df.groupby("wine_type")["wine_type"].count() / len(df) *100
sns.barplot(x= lst, y=(df["wine_type"].value_counts()/len(df)*100), data=df)
plt.bar(,)
lst = ["Portugal White Blend", "Albarino", "Loureiro","Arinto-Loureiro"]
df["wine_type"].unique()
df["wine_type"].value_counts()
sns.barplot(x='quality',y=df.groupby("wine_type")["wine_type"].count())

sns.barplot(x= df["quality"].unique() , y=(df["quality"].value_counts()/len(df)*100))

plt.bar(df[])
df["quality"].
sns.barplot(x= df["quality"].unique() , y=(df["quality"].value_counts()/len(df)*100))
