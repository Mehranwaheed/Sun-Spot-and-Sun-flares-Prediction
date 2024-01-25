# Project Overview
The Sunspot and Sunflare Prediction project focuses on forecasting severe solar flares, specifically X-class and M-class flares, using data related to sunspots and solar flares. Sunspots, dark areas on the sun's surface, are known to be associated with solar flares, and understanding their behavior is crucial for predicting potentially disruptive solar activity.
# Project objectives
**Data Collection**

**Exploratory Data Analysis (EDA)**

**Preprocessing**

**Feature Engineering**

**Modeling**

**Evaluation**

**Hyperparameter Tuning**

**Interpretation and Insights**

# Sun Spots
importing Necessary libraries

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
```

# Preprocessing
Sunspots Dataset loading into Pandas Dataframe
```python
df_spot=pd.read_excel("/content/drive/MyDrive/Datasets/sunspot_data.xlsx",sheet_name="spots1981-2017")
df_spot.drop(columns=["Unknown_var"],inplace=True)
df_spot.sample(5)
```
![image](https://github.com/Mehranwaheed/Sun-Spot-and-Sun-flares-Prediction/assets/119947085/8f4f2035-5449-42eb-85bb-6340d5a07203)

It is evident that the dataset contains a substantial amount of missing values.

```python
df_spot.info()
```
![image](https://github.com/Mehranwaheed/Sun-Spot-and-Sun-flares-Prediction/assets/119947085/f89bc697-664c-4653-a82d-52a8b979dc0a)

The accuracy of data types is not precise.

first, we have to format the date and time into date time format. in date column it contains ID as well so have to split it to get date.
```python
df_spot_copy=df_spot.copy()
```
Generated a duplicate of the original dataset for further analysis, following good practice guidelines.
```python
df_spot_copy['Date']=df_spot_copy['Date'].astype(int)
df_spot_copy['Date']=df_spot_copy['Date'].astype(str)
df_spot_copy['Date']=df_spot_copy['Date'].str[2:]
df_spot_copy['Date']=pd.to_datetime(df_spot_copy['Date'],format="%y%m%d",errors='coerce')
df_spot_copy.sample(5)
```
![image](https://github.com/Mehranwaheed/Sun-Spot-and-Sun-flares-Prediction/assets/119947085/e1477123-774e-4377-82b1-08be10013297)
now have to convert time into date time format.
```python
df_spot_copy['Time']=df_spot_copy['Time'].astype(int)
df_spot_copy['Time']=df_spot_copy['Time'].astype(str)
df_spot_copy['Time']=pd.to_datetime(df_spot_copy['Time'],format="%H%M",errors="coerce").dt.time
df_spot_copy.sample(5)
```
![image](https://github.com/Mehranwaheed/Sun-Spot-and-Sun-flares-Prediction/assets/119947085/de151f88-4641-4bc6-8181-0c66f07e9213)

### checking for null values
```python
df_spot_copy.isna().sum()
```
![image](https://github.com/Mehranwaheed/Sun-Spot-and-Sun-flares-Prediction/assets/119947085/97c43fb8-9d7d-457a-b061-e16988ee71f2)
we have seen there are null values in almost every feature but null values in "Date", "Time", "Region_number" can't be imputed so have to drop the null values and null values in "Number of sunspots ","length" and "area" can be imputed by filling the mean value. dropping the features such as "individual date","regional date","station_number","observartories","location", "time" and "Mount Wilson Class" as they are not conveying any useful information.
```python
model_df=df_spot_copy.copy()
model_df.drop(columns=["Time","Location","Mount Wilson Class","individual date","regional date","station_number","observartories","Region_number"],inplace=True)
model_df.dropna(subset=["Date"],inplace=True)
model_df['Number of sunspots']=model_df['Number of sunspots'].fillna(model_df['Number of sunspots'].mean())
model_df['length']=model_df['length'].fillna(model_df['length'].mean())
model_df['area']=model_df['area'].fillna(model_df['area'].mean())
model_df.isna().sum()
```
![image](https://github.com/Mehranwaheed/Sun-Spot-and-Sun-flares-Prediction/assets/119947085/2c496100-ea0b-4563-9459-2d8cf05f0237)

# Exploratory Data Analysis
```python
plt.figure(figsize=(28,6))
plt.plot(model_df['Date'],model_df["Number of sunspots"])
```
![image](https://github.com/Mehranwaheed/Sun-Spot-and-Sun-flares-Prediction/assets/119947085/e6c14f42-7a74-418f-aaed-c391d82ce9f1)
we can see the pattern of solar cycle.
To delve deeper into the sunspot patterns, transformed the sunspot count into the monthly mean number.

```python
plt.figure(figsize=(28,6))
plt.plot(monthly_mean.index, monthly_mean, label='Monthly Mean Sunspots', linestyle='-', color='blue')
plt.ylabel("sunspots number mean")
plt.xlabel("years")
plt.title("Monthly mean number of Sunspots")
```
![image](https://github.com/Mehranwaheed/Sun-Spot-and-Sun-flares-Prediction/assets/119947085/b2d53711-d438-46a3-8525-6d4970c44d9a)
The solar flare cycle is distinctly evident in this figure.
Observing this, we recognize the significance of sunspot numbers. Hence, we aim to forecast the solar cycle by predicting the number of sunspots. High sunspot numbers indicate the solar cycle maximum, whereas low numbers signify the solar cycle minimum.

### train test split
```python
X = df.drop("value", axis=1)  # Features
y = df["value"]  # Target variable
X_train=X.iloc[0:240,0:5]
X_test=X.iloc[241:,0:5]
y_train=y.iloc[0:240]
y_test=y.iloc[241:]
```
### XG Boost
```python
model = xgb.XGBRegressor(n_estimators=110, max_depth=2, learning_rate=0.1, early_stopping_rounds=3)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
predictions = model.predict(X_test)
# Evaluate performance
mae = mean_absolute_error(y_test, predictions)
print("MAE:", mae)
```
**MAE:** 2.0519231748071673

```python
plt.figure(figsize=(28,6))
plt.plot(df.index,df['value'],label="Actual",linestyle="dotted")
plt.plot(X_test.index,predictions,label="predicted",linestyle="-")
plt.title("Predictions by XgBoost")
plt.legend()
```
![image](https://github.com/Mehranwaheed/Sun-Spot-and-Sun-flares-Prediction/assets/119947085/c39a7212-aab3-4f9e-9b0a-d8d9d0a00f98)

### Decision Tree Regressor
```python
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(random_state=42)

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f'Mean absolute Error: {mae}')
```
**Mean absolute Error:** 2.69008801516849

```python
plt.figure(figsize=(28,6))
plt.plot(df.index,df['value'],label="Actual",linestyle="dotted")
plt.plot(X_test.index,predictions,label="predicted",linestyle="-")
plt.title("Predictions by DecisionTreeRegressor")
plt.ylabel("Monthly mean Sunspot Numbers")
plt.xlabel("Years")

plt.legend()
```
![image](https://github.com/Mehranwaheed/Sun-Spot-and-Sun-flares-Prediction/assets/119947085/ef2378da-8933-42d1-ac9a-69a8f88c61c5)

**To conduct further analysis, merge the sunspot dataset with the sunflare dataset. However, as a preliminary step, the flare dataset needs cleaning. We will address this aspect first and return to the analysis later.**

# SunFlares
Following the same approach as mentioned earlier, loaded the dataset into a pandas dataframe and ensured the correct conversion of the date and time columns to their respective data types.
```python
falre_dataset.sample(5)
```
![image](https://github.com/Mehranwaheed/Sun-Spot-and-Sun-flares-Prediction/assets/119947085/481a4fa9-f4ef-46ec-9f87-579d7fabe01f)

after exploring the flare dataset observed that xray flux had missing values before 1996. so to plot xray flux created the mask.
```python
filtered_dataframe=falre_dataset[falre_dataset['year']>1996]
plt.figure(figsize=(20,8))
plt.plot(filtered_dataframe['Date'],filtered_dataframe['xray_flux'])
plt.xlabel("Years")
plt.ylabel("Xray_flux")
plt.yscale("log")
```
![download (8)](https://github.com/Mehranwaheed/Sun-Spot-and-Sun-flares-Prediction/assets/119947085/3737653a-9fc4-43e0-ad43-91b18741febe)
from this figure it is observed that number of sunspot and xray flux follows the same pattern it means the most powerful flares occur at the peak of solar cycle and at that time number of sunspots are high as well.

```python

occurrences_df = falre_dataset.groupby('classification').size().reset_index(name='occurrences')


flux_df = falre_dataset.groupby('classification')['xray_flux'].mean().reset_index(name='mean_xray_flux')


merged_df = pd.merge(occurrences_df, flux_df, on='classification')


print(merged_df)


fig, axes = plt.subplots(2, 1, figsize=(8, 8))

sns.barplot(data=occurrences_df, x='classification', y='occurrences', ax=axes[0])
axes[0].set_title('Occurrences of Flares by Classification')
axes[0].set_yscale("log")
sns.barplot(data=merged_df, x='classification', y='mean_xray_flux', ax=axes[1])
axes[1].set_title('Mean X-ray Flux by Classification')
axes[1].set_yscale("log")
plt.tight_layout()

plt.show()
```
![download (2)](https://github.com/Mehranwaheed/Sun-Spot-and-Sun-flares-Prediction/assets/119947085/f31db315-4e39-4922-822c-d42057dd7dff)

We observed flare classifications, namely A, B, C, M, and X, with the majority of occurrences being of B and C class flares, while X class flares were less frequent and X class flares are most powerful flares.

After exploring both the data set it's observed that NOAA or Region number is the common column so would be able to merge through it.

```python
df_flare.head()
```
![image](https://github.com/Mehranwaheed/Sun-Spot-and-Sun-flares-Prediction/assets/119947085/f0768799-28eb-4722-8a86-ad657da23352)

```python
df_sunspot.head()
```
![image](https://github.com/Mehranwaheed/Sun-Spot-and-Sun-flares-Prediction/assets/119947085/619bc7ad-ca68-481d-9636-645c1a2e8ca6)

```python
merged_df = pd.merge(df_sunspot, df_flare,on='Date',how='left')
merged_df.info()

```
![image](https://github.com/Mehranwaheed/Sun-Spot-and-Sun-flares-Prediction/assets/119947085/8b436251-986c-4d08-a107-30f88e835c5e)

### Dealing with Null Values
```python
merged_df.isna().sum()
```
![image](https://github.com/Mehranwaheed/Sun-Spot-and-Sun-flares-Prediction/assets/119947085/86778994-a832-4585-9278-b495bce7cef7)

can't impute null values in time and region number so decided to drop them
```python
merged_df.dropna(subset=["start_time"],inplace=True)
merged_df.dropna(subset=["Region_number"],inplace=True)
merged_df.isna().sum()
```
![image](https://github.com/Mehranwaheed/Sun-Spot-and-Sun-flares-Prediction/assets/119947085/63f7c741-9345-46ed-900e-e2b97bfec674)

Qahwaji, R. and Colak, T. (2007) Automatic Short-Term Solar Flare Prediction Using Machine Learning 
and Sunspot Associations. Solar Physics 241 (1), 195–211
from this research, found out the rules to create association between sunspot and sun flares data set rules are:
 Find the time difference between the flare eruption time and the extracted sunspots
time.
• If the time difference is less than six hours then the flare record and the sunspot
record are marked as being ASSOCIATED.
• If there is more than one classification report for the flare-associated sunspot group
within six hours then choose the one with minimum time difference.
• Get the sunspot groups that are not associated with any flare.
– If no solar flares anywhere on the Sun occur within one day after the classification of
this sunspot group then it is marked as UNASSOCIATED.
• Record all ASSOCIATED groups and their corresponding flare classes (an example of
which is given in Figure 1).
• Record UNASSOCIATED groups.
• Create the data sets using McIntosh class

so, to calculate the time used the custom function

```python
from datetime import datetime

def calculate_time_difference(g):
    try:
        # Convert datetime strings to datetime objects
        start_datetime = datetime.strptime(str(g["Time"]), "%Y-%m-%d %H:%M:%S")
        end_datetime = datetime.strptime(str(g["start_time"]), "%Y-%m-%d %H:%M:%S")

        # Calculate the time difference
        time_difference = end_datetime - start_datetime
        if time_difference.total_seconds() < 0:
          return start_datetime - end_datetime
        else:
          return time_difference

            #time_difference = start_datetime - end_datetime




    except ValueError as e:
        return f"Error: {e}"
def time(f):
  return f["Date_time_f"]-f["Date_time_s"]

merged_df["time_difference_1"]=merged_df.apply(time,axis=1)
merged_df["time_difference"]=merged_df.apply(calculate_time_difference,axis=1)


```
after calculating the time difference next step is to match the region number to create associated and unassociated data

```python
def check_number_match(d):
    reference_str = str(d["NOAA"])
    target_str = str(d["Region_number"])

    # Check if the numbers completely match
    if reference_str == target_str:
        return "matched"

    # Check if the first 5 digits match
    elif reference_str[1:] == target_str[:]:
        return "matched"
    elif reference_str[2:] == target_str[:]:
        return "matched"

    # If neither condition is met
    else:
        return "Not matched"
filtered_df["NOAA_check"]=filtered_df.apply(check_number_match,axis=1)
filtered_df1=filtered_df[filtered_df["NOAA_check"]=="matched"]
```
masked the data to use the associated sunspot and sunflares dataset for further analysis also as explained above interested in only M and X class flares so masked out other flare types.
```python
filtered_df1=filtered_df1[filtered_df1['classification'].isin(["M", "X"])]
```
as in rules after creating the association to create initial dataset for model have to convert MCIntosh classes into Numberical form according to the method explained in the research paper.
```python
def striping(f):
  return f["McIntosh_class"].strip()
filtered_df1["McIntosh classes"]=filtered_df1.apply(striping, axis=1)
filtered_df1["McIntosh classes"]= filtered_df1["McIntosh classes"].replace('', pd.NA)
filtered_df1[['Class1', 'Class2',"Class3","class4","class5","class6"]]=filtered_df1['McIntosh classes'].str.split("",expand=True)
filtered_df1.drop(columns=["Class1","class5","class6"],inplace=True)
class_1={"A":0.10,"H":0.15,"B":0.30,"C":0.45,"D":0.60,"E":0.75,"F":0.90}
def map_class_1(class_1_value):
    return class_1.get(class_1_value, pd.NA)

filtered_df1["Numeric_class1"]=filtered_df1['Class2'].apply(map_class_1)

class_2={"X":0,"R":0.10,"S":0.30,"A":0.50,"H":0.70,"K":0.90}
def map_class_2(class_2_value):
    return class_2.get(class_2_value, pd.NA)

filtered_df1["Numeric_class2"]=filtered_df1['Class3'].apply(map_class_2)

class_3={"X":0,"O":0.10,"C":0.90,"I":0.50}
def map_class_3(class_3_value):
    return class_3.get(class_3_value, pd.NA)

filtered_df1["Numeric_class3"]=filtered_df1['class4'].apply(map_class_3)

class_2={"X":1,"R":2,"S":3,"A":4,"H":5,"K":6}
def map_class_2(f):
  return class_2.get(f, pd.NA)
filtered_df1["calc"]=filtered_df1['Class3'].apply(map_class_2)

def p_component(f):
  return f["calc"]/6
filtered_df1["p_component"]=filtered_df1.apply(p_component,axis=1)

def normalize(f):
  return f["area"]/1000
filtered_df1["normalized_area"]=filtered_df1.apply(normalize,axis=1)

def calculation(f):
  x=abs((f["normalized_area"]-f["p_component"])*f["normalized_area"])
  return x
filtered_df1["fourth_comp"]=filtered_df1.apply(calculation,axis=1)
```
normalized the sunspot number,length and area in this merged data set

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
column_to_scale = 'Number of sunspots'

# Extract the column as a 2D array (reshape if needed)
column_data = filtered_df1[column_to_scale].values.reshape(-1, 1)
scaled_column = scaler.fit_transform(column_data)

# Update the DataFrame with the scaled values
filtered_df1[column_to_scale] = abs(scaled_column)
filtered_df1.sample(5)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
column_to_scale = 'area'

# Extract the column as a 2D array (reshape if needed)
column_data = filtered_df1[column_to_scale].values.reshape(-1, 1)
scaled_column = scaler.fit_transform(column_data)

# Update the DataFrame with the scaled values
filtered_df1[column_to_scale] = abs(scaled_column)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
column_to_scale = 'length'

# Extract the column as a 2D array (reshape if needed)
column_data = filtered_df1[column_to_scale].values.reshape(-1, 1)
scaled_column = scaler.fit_transform(column_data)

# Update the DataFrame with the scaled values
filtered_df1[column_to_scale] = abs(scaled_column)
```
dropped the unnecessary columns
```python
filtered_df1.drop(columns=['Unnamed: 0','Time', 'Mount Wilson Class', 'Region_number',
                           'McIntosh_class','Date_time_s','start_time', 'end_time', 'peak_time',
                           'type', 'NOAA','Classification+type', 'Date_time_f', 'time_difference_1',
                           'time_difference', 'NOAA_check', 'McIntosh classes','Class2', 'Class3',
                           'class4','calc','p_component'
                           ],inplace=True)    #,'xray_flux'
```
converted M and X into 0 and 1 as machine learning algorithms performed well with numerical values.
```python
filtered_df1['classification'] = filtered_df1['classification'].map({'M': 0, 'X': 1})
filtered_df1.info()
```
final data set have following columns

![image](https://github.com/Mehranwaheed/Sun-Spot-and-Sun-flares-Prediction/assets/119947085/0d8fc274-00cc-4626-b2dc-0c82da1cdaa7)

Data set have only two classes 0 and 1, after exploring it it's found out that data is highly imbalanced as X class flares are just 7.2% and M class flares are 92.8%

![download (9)](https://github.com/Mehranwaheed/Sun-Spot-and-Sun-flares-Prediction/assets/119947085/7080d26d-b6f7-4d04-882e-ce0d3c24c57f)

# Train Test Split
splitted the data by keeping the same ratio in both testing and training set

```python
X = filtered_df1.drop(columns=['classification',"xray_flux"], axis=1)
y = filtered_df1['classification']
round(X.shape[0]*0.80,0)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
# Define a dataframe containing frequency percentages
df_perc = pd.concat([y.value_counts(normalize=True).mul(100).round(1),
                     y_train.value_counts(normalize=True).mul(100).round(1),
                     y_test.value_counts(normalize=True).mul(100).round(1)], axis=1)
df_perc.columns=['Dataset','Training','Test']
df_perc = df_perc.T

# Plot frequency percentages barplot
df_perc.plot(kind='barh', stacked=True, figsize=(10,5), width=0.6)

# Add the percentages to our plot
for idx, val in enumerate([*df_perc.index.values]):
    for (percentage, y_location) in zip(df_perc.loc[val], df_perc.loc[val].cumsum()):
        plt.text(x=(y_location - percentage) + (percentage / 2)-3,
                 y=idx - 0.05,
                 s=f'{percentage}%',
                 color="black",
                 fontsize=12,
                 fontweight="bold")
```
used 80% of data for training and 20% for testing the results

![download (14)](https://github.com/Mehranwaheed/Sun-Spot-and-Sun-flares-Prediction/assets/119947085/49a66cfe-a9be-44a6-a8b7-b840433a7ac8)

As dataset is highly imbalanced so need to balance it but before doing the data balancing just testing the results by using Random forest classifier.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


#class_weight=dict({"0.1":1,"0.9":1000})
classifier=RandomForestClassifier()
grid={"criterion":["gini","entropy","log_loss"],"max_depth":[1,10,20,100],"max_features":[1,2,3,4,5,6,7]}
cv=KFold(n_splits=5,random_state=None,shuffle=False)
classifier=GridSearchCV(classifier,grid,cv=cv,n_jobs=-1,scoring="f1_macro")

classifier.fit(X_train,y_train)
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

y_pred=classifier.predict(X_test)
#print(confusion_matrix(y_test,y_pred))
print(f"Accuracy Socre: {round(accuracy_score(y_test,y_pred)*100,2)}%")
print("---------------------------------------------------------")
print("Classification Report")
print(classification_report(y_test,y_pred))
print("---------------------------------------------------------")
print("Confusion Matrix")
print(confusion_matrix(y_test,y_pred))
```

![Screenshot 2024-01-25 170558](https://github.com/Mehranwaheed/Sun-Spot-and-Sun-flares-Prediction/assets/119947085/de6477a9-21da-4929-8a99-67ab974ec669)

While achieving an accuracy above 80%, it's crucial to recognize that this metric may not present an accurate picture, especially in the context of imbalanced datasets. The confusion matrix exposes a notable imbalance, specifically with the classifier correctly classifying only 29 out of 211 X class flares. This underscores the importance of considering other evaluation metrics and highlights the limitation of relying solely on accuracy in imbalanced datasets.

# Addressing Imbalanced Data Problem

## SMOTomek

```python
from imblearn.combine import SMOTETomek
st=SMOTETomek()
X_train_st,y_train_st=st.fit_resample(X_train,y_train)
```
Retrained the model with a balanced dataset, shifting the focus to the F1 score as the accuracy metric. This choice was made for its balance between precision and recall. Notably, emphasizing better precision was prioritized to enhance the accuracy of predictions for X class flares.

```python
from sklearn.ensemble import RandomForestClassifier
#class_weight=dict({"0.1":1,"0.9":1000})
classifier=RandomForestClassifier()
grid={"criterion":["gini","entropy","log_loss"],"max_depth":[1,10,20,100],"max_features":["sqrt","log","None"]}
cv=KFold(n_splits=5,random_state=None,shuffle=False)
classifier=GridSearchCV(classifier,grid,cv=cv,n_jobs=-1,scoring="f1_macro")

classifier.fit(X_train_st,y_train_st)

y_pred=classifier.predict(X_test)
print(f"Accuracy Score: {round(accuracy_score(y_test,y_pred)*100,2)}%")
print("----------------------------------------------------------")
print("Classification Report")
print(classification_report(y_test,y_pred))
print("----------------------------------------------------------")

print(confusion_matrix(y_test,y_pred))
```
![Screenshot 2024-01-25 172622](https://github.com/Mehranwaheed/Sun-Spot-and-Sun-flares-Prediction/assets/119947085/4bc75bcf-6684-4159-9616-4a6a64f6cb8d)

As the results remained unsatisfactory, the decision was made to explore the precision-recall tradeoff.
The precision-recall tradeoff is a critical concept in machine learning that involves finding a balance between precision and recall when adjusting the classification threshold for a model. Precision and recall are two important metrics used to evaluate the performance of a classifier, especially in situations where class imbalance exists.

Precision: Precision measures the accuracy of positive predictions, indicating the proportion of correctly predicted positive instances out of all instances predicted as positive. It is calculated as the ratio of true positives to the sum of true positives and false positives.

Recall (Sensitivity): Recall, also known as sensitivity or true positive rate, measures the ability of a classifier to identify all positive instances. It is calculated as the ratio of true positives to the sum of true positives and false negatives.

The precision-recall tradeoff occurs when adjusting the classification threshold. As the threshold increases, precision typically improves while recall decreases, and vice versa. A higher threshold leads to a more conservative classifier that makes fewer positive predictions, potentially improving precision but reducing recall. Conversely, a lower threshold increases positive predictions, enhancing recall but potentially reducing precision.

must consider the application's specific requirements when navigating the precision-recall tradeoff. In scenarios where false positives and false negatives have varying costs, finding the optimal threshold becomes crucial. Balancing precision and recall is essential for developing a model that meets the desired performance criteria.

```python
import numpy as np
def check_(y_prob,value):
# Assuming y_scores is the array of predicted probabilities
  threshold = value

# Apply thresholding to get the predicted labels
  predicted_labels = np.where(y_prob[:, 1] > threshold, 1, 0)
  y_pred=classifier.predict(X_test)
  print(confusion_matrix(y_test,predicted_labels))
  print(accuracy_score(y_test,predicted_labels))
  print(classification_report(y_test,predicted_labels))
def tradeoff(y_prob):
  pr_list=[]
  re_list=[]
  thresh=[]
  for i in range(-70,100,1):
    value = i / 100.0
    thresh.append(value)

    import numpy as np

# Assuming y_scores is the array of predicted probabilities
    threshold = value

# Apply thresholding to get the predicted labels
    predicted_labels = np.where(y_prob[:, 1] > threshold, 1, 0)

# Now 'predicted_labels' contains the binary predictions based on the threshold
    predicted_labels
    pr = precision_score(
      y_test,
      predicted_labels,
      labels=None,
      pos_label=1,
      average='binary',
      sample_weight=None,
      zero_division='warn',
                 )
    pr_list.append(pr)
    re = recall_score(
        y_test,
        predicted_labels,
        labels=None,
        pos_label=1,
        average='binary',
        sample_weight=None,
        zero_division='warn',
       )
    re_list.append(re)

  return pr_list,re_list,thresh

def plot_(thresh,re_list,pr_list):
  plt.figure(figsize=(8,6))

  plt.plot(thresh,re_list,label="recall",color="red")
  plt.plot(thresh,pr_list,label="precision",color="green")
#plt.axvline(x=0.53, color='blue', linestyle='--', label='balanced')
#plt.text(0.53, 0.4, '0.53', color='blue', rotation=45, va='baseline',size="large",style="oblique")

#plt.axvline(x=0.60, color='black', linestyle='--', label='best Precision')
#plt.axvline(x=0.30, color='lime', linestyle='--', label='best Recall')
  plt.xlabel("Threshold Value")
  plt.title("Precision/Recall Tradeoff")
  plt.legend()
  plt.show()
precision_list , recall_list, thresh=tradeoff(y_prob)
plot_(thresh,recall_list,precision_list)
```
![download (8)](https://github.com/Mehranwaheed/Sun-Spot-and-Sun-flares-Prediction/assets/119947085/e9e7a826-969e-462c-88a6-c6ecb502ecd5)

```python
check_(y_prob,0.09)
```
![Screenshot 2024-01-25 173336](https://github.com/Mehranwaheed/Sun-Spot-and-Sun-flares-Prediction/assets/119947085/271bcae3-b84b-4b57-9630-365c8fb05134)


By elevating the threshold value, it becomes evident that better predictions for X class flares can be achieved. Despite a decrease in the overall accuracy score, the precision score notably improves.

# SVM Classsifier

```python
from sklearn.pipeline import Pipeline
import seaborn as sns
from sklearn.svm import SVC
from imblearn.combine import SMOTETomek
#st=SMOTETomek()
clf = SVC(kernel="rbf", random_state=42,C=1,gamma=10,probability=True)
#pipe=Pipeline([('sampling', SMOTETomek(sampling_strategy='auto')),('classification', RandomForestClassifier())])
clf.fit(X_train_st,y_train_st)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy : {accuracy:.2f}')
plt.figure(figsize=(8, 6))
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d",  xticklabels=['M', 'X'], yticklabels=['M', 'X'],cbar=False,cmap="cividis")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()
```
![download (19)](https://github.com/Mehranwaheed/Sun-Spot-and-Sun-flares-Prediction/assets/119947085/8dda5c3a-df52-4785-8c1e-cad054550565)


Utilizing the precision-recall tradeoff approach, improved results were obtained with Support Vector Machines (SVM). This technique involved adjusting the classification threshold, and the SVM model demonstrated enhanced performance, showcasing the effectiveness of fine-tuning the threshold for specific requirements.

![Screenshot 2024-01-25 174419](https://github.com/Mehranwaheed/Sun-Spot-and-Sun-flares-Prediction/assets/119947085/29fe64bf-21da-4a56-95ee-aa03b5eb277e)

# Conclusion

In conclusion, this project aimed to predict severe solar flares, specifically X-class flares, through a comprehensive analysis of sunspot and solar flare datasets. The initial exploration revealed challenges related to imbalanced data, particularly in the infrequent occurrence of X-class flares.

Efforts were made to address this imbalance through various strategies, including retraining models with balanced datasets, shifting to the F1 score as the evaluation metric, and exploring the precision-recall tradeoff. The precision-recall tradeoff proved to be a valuable technique, especially when applied to Support Vector Machines (SVM), leading to notable improvements in prediction accuracy for X-class flares.

It's essential to acknowledge the limitations and complexities associated with solar flare prediction, given the dynamic and intricate nature of solar activity. Continuous refinement and exploration of various methodologies, including those discussed in this project, are imperative for advancing the accuracy and reliability of solar flare prediction models.

As we continue to refine our understanding and methodologies for predicting solar flares, the insights gained from this project contribute to the ongoing efforts to enhance space weather forecasting capabilities, with potential implications for various technological systems affected by solar activity.

