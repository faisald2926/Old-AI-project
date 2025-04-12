
import pandas as pd
df = pd.read_csv("brain_stroke2.csv")



df.info()

df.isnull().sum()

df.head()

df.tail()

df.head(20)

df.columns

df['ever_married'].unique()

df.nunique()

df["smoking_status"].unique()

df.nunique()

df.head()

df.head()

from sklearn.preprocessing import LabelEncoder


df_new = df.copy()


le = LabelEncoder()

for column in df_new.columns:
    if df_new[column].dtype == type(object):
        df_new[column] = le.fit_transform(df_new[column])

df_new.isnull().sum()

df_new.head()

from sklearn.model_selection import train_test_split

X = df_new.drop(['stroke'], axis=1)
y = df_new['stroke']
X_train , X_test , y_train , y_test = train_test_split(X, y, test_size=0.2,random_state=4,shuffle=True)

from sklearn.linear_model import LogisticRegression

def linear_model(X_train,y_train,X_test):
  LR=LogisticRegression(max_iter=1000)
  LR.fit(X_train,y_train)
  y_pred=LR.predict(X_test)
  return y_pred

columns_to_keep = ['gender', 'age', 'ever_married', 'avg_glucose_level', 'bmi']


df_new = df_new[columns_to_keep]


model = LogisticRegression()
model.fit(X_train[columns_to_keep], y_train)


importance = model.coef_[0]


feature_importance = pd.DataFrame({'feature': df_new.columns, 'importance': importance})

feature_importance.sort_values('importance').plot(kind='barh', x='feature', y='importance')
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.show()

from sklearn.metrics import accuracy_score


y_pred=linear_model(X_train,y_train,X_test)
print('Logistic Regression Model :', accuracy_score(y_test,y_pred))
