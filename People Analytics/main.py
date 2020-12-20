import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
# Metrik evaluasi yang akan digunakan adalah AUC (Area Under ROC Curve), 
# dimana nilai True Positive Rate akan dibandingkan dengan nilai False Positive 
# Rate dalam threshold yang berbeda-beda.
# Gunakan AUC untuk evaluasi metrix
plt.figure(figsize=(12,7))

df_test= pd.read_csv("test.csv")
df = pd.read_csv("train.csv") #dataset untuk ditrain, include label

# print(df_test.info())
df = df[df['Employee_type'].notna()]
# df = df.fillna(df.mean())
df['achievement_target_3'] = df['achievement_target_3'].replace('not_reached','not reached')
df['achievement_target_2'] = df['achievement_target_2'].replace('Pencapaian < 50%','achiev_< 50%')
df['achievement_target_2'] = df['achievement_target_2'].replace("Pencapaian 50%-100%",'achiev_50%-100%')
df['achievement_target_2'] = df['achievement_target_2'].replace("Pencapaian 100%-150%",'achiev_100%-150%')
df['achievement_target_2'] = df['achievement_target_2'].replace("Pencapaian > 1.5",'achiev_> 1.5')
df['achievement_target_1'] = df['achievement_target_1'].replace('Pencapaian < 50%','achiev_< 50%')
df['achievement_target_1'] = df['achievement_target_1'].replace("Pencapaian 50%-100%",'achiev_50%-100%')
df['achievement_target_1'] = df['achievement_target_1'].replace("Pencapaian 100%-150%",'achiev_100%-150%')
df['achievement_target_1'] = df['achievement_target_1'].replace("Pencapaian > 1.5",'achiev_> 1.5')
print(df.isnull().sum())

# merubah object menjadi category column
obj_columns = df.select_dtypes(['object']).columns
df[obj_columns] = df[obj_columns].astype('category')
#merubah category menjadi int column
cat_columns = df.select_dtypes(['category']).columns
df[cat_columns] = df[cat_columns].apply(lambda x:x.cat.codes)

#coba-coba model

X = df.drop('Best Performance',axis=1)
y = df['Best Performance']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
#Decision Tree
# from sklearn.tree import DecisionTreeClassifier
# dtree = DecisionTreeClassifier()
# dtree.fit(X_train,y_train)
# pred = dtree.predict(X_test)

#Logistic Regression
# from sklearn.linear_model import LogisticRegression
# lr = LogisticRegression()
# lr.fit(X_train,y_train)
# pred = lr.predict(X_test)

#KNN
# error_rate = []
# from sklearn.neighbors import KNeighborsClassifier
# for i in range(1,40):
#     knn = KNeighborsClassifier(n_neighbors=i)
#     knn.fit(X_train,y_train)
#     pred_i = knn.predict(X_test)
#     error_rate.append(np.mean(pred_i!=y_test))

# plt.figure(figsize=(10,6))
# plt.plot(range(1,40),error_rate,color='blue',linestyle='dashed',marker='o',markerfacecolor='red',markersize=10)
# plt.title('Error Rate vs K Value')
# plt.xlabel('K')
# plt.ylabel('Error Rate')
# plt.show()
# knn = KNeighborsClassifier(n_neighbors=10)
# knn.fit(X_train,y_train)
# pred = knn.predict(X_test)
# from sklearn.metrics import roc_auc_score,confusion_matrix,classification_report

# print(roc_auc_score(pred,y_test))
# print(confusion_matrix(pred,y_test))
# print(classification_report(pred,y_test))


# plt.show()


# print(df.isnull().sum())
#Classification Algorithm = Logistic Regression, KNN, Tree Methods, 
# plt.show()

#KETERANGAN
#features yang bolong adalah 
# achievement_target_3 = status pencapaian target kategori 3
#achievement_target_2 = status pencapaian target kategori 2
#achievement_target_1 = status pencapaian target kategori 1
#Achievment_above_100%_during3quartal = Jumlah pencapaian diatas 100% dalam 3 tahun terkahir
#Last_Achievment_% = presentase pencapaian triwulan terakhir terhadap target
#Avg-achievment_% = rata - rata presentase pencapaian terhadap target selama setahun
#Education level = Tingkat pendidikan (0: Internal course/sem, 1: SLTA/Setingkat, 2: Diploma 1, 3: Diploma 3/4, 4: Strata1, 5:Strata2)
#GPA = IPK
#year_graduated = Tahun lulus
#job_duration_as_permanent_worker = lama bekerja sebaga pekerja tetap
#Employee_status

#Data Exploration and Cleaning
# print(df[['achievment_target_1','achievment_target_2','achievment_target_3']].tail())
# print(sns.heatmap(df[['achievment_target_3','achievment_target_2','achievment_target_1']],cbar=False,cmap="viridis",annot=True))
# print(df[['achievement_target_1','achievement_target_2','achievement_target_3']].mean())
# print(sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis'))
# print(sns.heatmap(df.corr(),cmap="coolwarm",annot=True))

