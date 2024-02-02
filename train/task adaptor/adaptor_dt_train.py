import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


df = pd.read_csv('./ton_test/raw.csv')
df = pd.read_csv('./ton_test/syn.csv')

validation_df = pd.read_csv('./ton_test/syn.csv')
validation_df = pd.read_csv('./ton_test/raw.csv')

features = ['ts', 'per_byt', 'srcip', 'dstip', 'srcport', 'dstport', 'proto']
label = 'type'
df_selected = df[features + [label]]
validation_df_selected = validation_df[features + [label]]


proto_encoder = LabelEncoder()
df_selected.loc[:, 'proto'] = proto_encoder.fit_transform(df_selected['proto'])
validation_df_selected.loc[:, 'proto'] = proto_encoder.transform(validation_df_selected['proto'])


X = df_selected[features]  
y = df_selected[label]     
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_validation = validation_df_selected[features]
y_validation = validation_df_selected[label]


dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

y_test_pred = dt.predict(X_test)
y_validation_pred = dt.predict(X_validation)


print("Internal Test Set Evaluation:\n", classification_report(y_test, y_test_pred))
print("Validation Set Evaluation:\n", classification_report(y_validation, y_validation_pred))
