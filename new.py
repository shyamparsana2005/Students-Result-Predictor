import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('ML\student_success_predictor\student_success_dataset.csv')

le = LabelEncoder()
df['Internet'] = le.fit_transform(df["Internet"]) #yes = 1, no = 0
df['Passed'] = le.fit_transform(df["Passed"])

features =  ['StudyHours', 'Attendance', 'PastScore', 'SleepHours']
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[features] = scaler.fit_transform(df[features])

#step - 4
x = df_scaled[features] #features
y = df_scaled['Passed'] #target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print('Classification Report : ')
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Fail", "Pass"], yticklabels=["Fail", "Pass"])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

print("------Predict Your Result------")

try:
    study_hours = float(input("Enter your study hours (0-24) : "))
    attendance = float(input("Enter your attendance (0-100) : "))
    past_score = float(input("Enter your past score (0-100) : "))
    sleep_hours = float(input("Enter your sleep hours (0-24): "))

    user_input_df = pd.DataFrame([{
        'StudyHours': study_hours,
        'Attendance': attendance,
        'PastScore': past_score,
        'SleepHours': sleep_hours
    }])

    user_input_scaled = scaler.transform(user_input_df)
    prediction = model.predict(user_input_scaled)[0]

    result = "Pass" if prediction == 1 else "Fail"
    print(f'Prediction Based on input: {result}')

except Exception as e:
    print('Error....', e)