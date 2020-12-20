import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("test.csv")
df['achievement_target_3'] = df['achievement_target_3'].replace('not_reached','not reached')
print(sns.countplot(df['achievement_target_3']))
plt.show()
# merubah object menjadi category column
obj_columns = df.select_dtypes(['object']).columns
df[obj_columns] = df[obj_columns].astype('category')
#merubah category menjadi int column
cat_columns = df.select_dtypes(['category']).columns
df[cat_columns] = df[cat_columns].apply(lambda x:x.cat.codes)

X = df.drop('achievement_target_3',axis=1)
y = df['achievement_target_3']



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
from sklearn.neighbors import KNeighborsClassifier

# error_rate = []
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

knn = KNeighborsClassifier(n_neighbors=27)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)

from sklearn.metrics import roc_auc_score,confusion_matrix,classification_report

print(roc_auc_score(pred,y_test))
print(confusion_matrix(pred,y_test))
print(classification_report(pred,y_test))

