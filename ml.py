import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.calibration import calibration_curve

import xgboost as xgb
import matplotlib.pyplot as plt
import shap
import joblib

DATA_PATH = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
df = pd.read_csv(DATA_PATH)

df.head()
df.columns
df.shape
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors= 'coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)
print(df.head())

df = df.drop('customerID', axis=1)

df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})
print(df['Churn'].value_counts(normalize=True))
# 0    0.73463 'No'
# 1    0.26537 'Yes'

#df['ChargePerTenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)

categorical_cols = df.select_dtypes(include='object').columns.tolist()
print(categorical_cols)

le_dict = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=42, stratify=y)
X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

X_test.to_csv('data/X_test.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)
joblib.dump(le_dict, 'data/le_dict.pkl')

scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1]) * 0.65

model = xgb.XGBClassifier(
    random_state=42,
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight,
    max_depth=4,          
    learning_rate=0.05,
    n_estimators=300,
    subsample=0.7,
    colsample_bytree=0.8,
    max_leaves=18,             
    min_child_weight=25,
    )

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]
print(classification_report(y_test, y_pred))
print("AUC-ROC:", roc_auc_score(y_test, y_proba))

joblib.dump(model, 'model/churn_model.pkl')

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)

prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0,1], [0,1], linestyle='--', label='Ideal')
plt.xlabel('Predicted Probability')
plt.ylabel('Actual Churn Rate')
plt.legend()
plt.show()

bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']

df_result = pd.DataFrame({'proba': y_proba, 'actual': y_test.values})
df_result['bin'] = pd.cut(df_result['proba'], bins=bins, labels=labels)

print(df_result.groupby('bin')['actual'].agg(['mean', 'count']))