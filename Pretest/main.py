import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#yang banyak null monthly income maka harus dibuat
#  age, DebtRatio, MonthlyIncome, NumberOfOpenCreditLinesAndLoans dan RevolvingUtilizationOfUnsecuredLines
plt.figure(figsize=(10,7))
df = pd.read_csv("credit-downsampled.csv")
print(df[df['Class'] == 0]['MonthlyIncome'].mean())
print(df[df['Class'] == 1]['MonthlyIncome'].mean())
print(df.info())
# print(sns.heatmap(df.corr(),cmap='coolwarm',annot=True))
def impute_income(cols):
    MonthlyIncome = cols[0]
    Class = cols[1]
    if pd.isnull(MonthlyIncome):
        if Class == 0:
            return 6740.5
        else:
            return 5631
    else:
        return MonthlyIncome
df['MonthlyIncome'] = df[['MonthlyIncome','Class']].apply(impute_income,axis=1)
# print(sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis'))
#DATA CLEARNING
# print(sns.boxplot(x='Class',y='MonthlyIncome',data=df))
# plt.show()

#MODEL
X= df[['age','DebtRatio','MonthlyIncome','NumberOfOpenCreditLinesAndLoans','RevolvingUtilizationOfUnsecuredLines']]
y= df['Class']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

from sklearn.linear_model import LogisticRegression
lrm = LogisticRegression()
lrm.fit(X_train, y_train)
predictions = lrm.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
print(sns.heatmap(df.corr(),cmap='coolwarm',annot=True))
plt.show()

