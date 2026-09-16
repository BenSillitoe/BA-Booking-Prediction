import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

df = pd.read_csv('customer_booking.csv', encoding= 'ISO-8859-1')

#Set up a 2x2 grid that shows us the raw histogram + log-transformed histogram for each column
#fig, axes = plt.subplots(2,2, figsize=(12,8))

# Purchase lead
#axes[0,0].hist(df['purchase_lead'], bins=50, color='red', edgecolor='black')
#axes[0,0].set_title('purchase_lead (raw)')
#axes[0,0].set_xlabel('Days before departure')

#axes[0,1].hist(np.log1p(df['purchase_lead']), bins=50, color='red', edgecolor='black')
#axes[0,1].set_title('purchase_lead (log-transformed)')

# Length of stay
#axes[1,0].hist(df['length_of_stay'], bins=50, color='blue', edgecolor='black')
#axes[1,0].set_title('length_of_stay (raw)')
#axes[1,0].set_xlabel('Days')

#axes[1,1].hist(np.log1p(df['length_of_stay']), bins=50, color='blue', edgecolor='black')
#axes[1,1].set_title('length_of_stay (log-transformed)')

#plt.tight_layout()
#plt.show()

#add log-transformed versions as new columns, so that the model can choose between raw and log data
df['purchase_lead_log'] = np.log1p(df['purchase_lead'])
df['length_of_stay_log'] = np.log1p(df['length_of_stay'])

#sanity check
#print(df[['purchase_lead', 'purchase_lead_log', 'length_of_stay', 'length_of_stay_log']].head())

#Frequancy encoding for high-cardinality routes
route_counts = df['route'].value_counts()
df['route_freq'] = df['route'].map(route_counts)

#One-hot ecoding for booking origin
df = pd.get_dummies(df, columns=['booking_origin', 'sales_channel', 'trip_type', 'flight_day']
                    , prefix=['origin', 'channel', 'trip', 'day'])

df = df.drop(columns=['route'])

X = df.drop(columns=['booking_complete'])
y = df['booking_complete']

X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size = 0.2, random_state = 42, stratify = y
)

#Training the random forest model
print(X_train.shape, X_test.shape)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state= 42,
    n_jobs = -1
)

model.fit(X_train, y_train)

#Get predictions for metrics
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

#First measure, we need to get the classification report
print(classification_report(y_test,y_pred))

#Then we need to see the confusion matrix, a raw count of right and rongs based on predictions made by the model
cm = confusion_matrix(y_test, y_pred)
print(cm)

#Lastly we need to see the ROC-AUC, an overall ability of the model to seperate the two classes
print('ROC-AUC:', roc_auc_score(y_test, y_proba))


#cross-validation using 5-fold stratified
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

roc_auc_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)

f1_scores = cross_val_score(model, X, y, cv=cv,scoring='f1', n_jobs=-1)

print('ROC-AUC per fold ', roc_auc_scores)
print(f"ROC-AUC mean: {roc_auc_scores.mean():.4f} (std: {roc_auc_scores.std():.4f}")

print('F1 per fold: ', f1_scores)
print(f"F1 mean: {f1_scores.mean():.4} (std: {f1_scores.std():.4f}")

#visualisation
importances = pd.Series(model.feature_importances_, index=X.columns)
top15 = importances. sort_values(ascending=False).head(15)

plt.figure(figsize=(12,8))
top15.sort_values().plot(kind='barh', color='blue')
plt.xlabel('Features Importance')
plt.title('Top 15 most important features - Random Forest Model', fontsize=13, pad=15)
plt.tight_layout()
plt.subplots_adjust(left=0.3)
plt.show()
