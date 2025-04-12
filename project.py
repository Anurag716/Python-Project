import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
import plotly.express as px

sns.set(style='whitegrid')
data=pd.read_csv("C:/Users/DELL/OneDrive/Desktop/project(Python)/project/oci_dataset_from08122005_to31122009.csv")
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
data=data.dropna(subset=['Date'])


#basic EDA prints
print("\n First 10 records of dataset")
print(data.head(10))
print("\n Last 10 records of dataset")
print(data.tail(10))
print("\n Statistic summary ")
print(data.describe())
print("\n summary information")
print(data.info())
print("\n Number of rows and columns(shape)")
print(data.shape)
print("Check for missing values")
print(data.isnull().sum())


df=data.drop(columns=['Country','Mission','Date','OCI - Enquiries'])

#correlation and covariance
print("\nCovariance:",df.cov())
corr=df.corr()
print("\nCorrelation:",df.corr())
#heatmap to visualize correlation between different columns of table
plt.figure(figsize=(13,7))
sns.heatmap(corr,annot=True,cmap='coolwarm',fmt='.2f',linewidths=0.5)
plt.title("\ncorrelation matrix")
plt.show()


#Outlier detection
#column-1
q1=np.percentile(df['OCI - Registered'],25)
q3=np.percentile(df['OCI - Registered'],75)
IQR =q3-q1
lower_bound= q1-1.5*IQR
upper_bound= q3+1.5*IQR
outliers_iqr = df[(df['OCI - Registered']<lower_bound)|(df['OCI - Registered']>upper_bound)]
print("\noutlier for ['OCI - Registered']:",outliers_iqr)
#column-2
q1=np.percentile(df['OCI - Issued'],25)
q3=np.percentile(df['OCI - Issued'],75)
IQR =q3-q1
lower_bound= q1-1.5*IQR
upper_bound= q3+1.5*IQR
outliers_iqr = df[(df['OCI - Issued']<lower_bound)|(df['OCI - Issued']>upper_bound)]
print("\noutlier for ['OCI - Issued']:",outliers_iqr)
#column-3
q1=np.percentile(df['Image - Scanned'],25)
q3=np.percentile(df['Image - Scanned'],75)
IQR =q3-q1
lower_bound= q1-1.5*IQR
upper_bound= q3+1.5*IQR
outliers_iqr = df[(df['Image - Scanned']<lower_bound)|(df['Image - Scanned']>upper_bound)]
print("\noutlier for ['Image - Scanned']:",outliers_iqr)
#column-4
q1=np.percentile(df['OCI - Granted'],25)
q3=np.percentile(df['OCI - Granted'],75)
IQR =q3-q1
lower_bound= q1-1.5*IQR
upper_bound= q3+1.5*IQR
outliers_iqr = df[(df['OCI - Granted']<lower_bound)|(df['OCI - Granted']>upper_bound)]
print("\noutlier for ['OCI - Granted']:",outliers_iqr)
#column-5
q1=np.percentile(df['OCI - Despatched to mission'],25)
q3=np.percentile(df['OCI - Despatched to mission'],75)
IQR =q3-q1
lower_bound= q1-1.5*IQR
upper_bound= q3+1.5*IQR
outliers_iqr = df[(df['OCI - Despatched to mission']<lower_bound)|(df['OCI - Despatched to mission']>upper_bound)]
print("\noutlier for ['OCI - Despatched to mission']:",outliers_iqr)

#boxplot visualization for outlier detection
plt.figure(figsize=(10, 6))
sns.boxplot(data=df)
plt.title("Boxplot of OCI Registered and Granted (Yearly Aggregated)")
plt.show()


#1st object:-   Top 10 countries with maximum OCI registration
plt.figure(figsize=(12, 6))
g=((data.groupby('Country',as_index=False)['OCI - Registered'].sum()).sort_values(by='OCI - Registered',ascending=False)).head(10)
sns.barplot(y=g['Country'],x=g['OCI - Registered'],errorbar=None)
plt.title("Top 10 Countries by OCI Registration")
plt.show()


#2nd objective:- how OCI-registration and OCI-Granted relatively change over years
pivot_table=data.pivot_table(values=['OCI - Registered','OCI - Granted'],index='Date',aggfunc='sum')
yearly_data=pivot_table.resample('YE').sum()
yearly_data = yearly_data.reset_index()
plt.figure(figsize=(10, 6))
sns.lineplot(data=yearly_data, x='Date', y='OCI - Registered', label='OCI - Registered')
sns.lineplot(data=yearly_data, x='Date', y='OCI - Granted', label='OCI - Granted')
plt.title("Yearly OCI Registered and Granted over Time")
plt.xlabel("Year")
plt.ylabel("Count")
plt.xticks(ticks=yearly_data['Date'], labels=yearly_data['Date'].dt.year)
plt.tight_layout()
plt.show()


#3rd objective:-  Top 10 missions with maximum OCI Registration
plt.figure(figsize=(12, 6))
g=((data.groupby('Mission',as_index=False)['OCI - Registered'].sum()).sort_values(by='OCI - Registered',ascending=False)).head(10)
sns.barplot(y=g['Mission'],x=g['OCI - Registered'],errorbar=None)
plt.title("Top 10 Mission by OCI Registration")
plt.show()


#4th objective:- how months or season affect registration,issuing and granting process
pivot_table=data.pivot_table(values=['OCI - Registered','OCI - Issued','OCI - Granted'],index='Date',aggfunc='sum')
monthly_data = pivot_table.resample('ME').sum()


monthly_data['Month'] = monthly_data.index.month_name().str[:3]
monthly_avg = monthly_data.groupby('Month').sum()

month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
monthly_avg = monthly_avg.loc[month_order]
plt.figure(figsize=(10, 6))
plt.plot(monthly_avg.index, monthly_avg['OCI - Registered'], label='Sum of OCI - Registered', color='orangered', marker='o')
plt.plot(monthly_avg.index, monthly_avg['OCI - Issued'], label='Sum of OCI - Issued', color='gold', marker='o')
plt.plot(monthly_avg.index, monthly_avg['OCI - Granted'], label='Sum of OCI - Granted', color='green', marker='o')
plt.title("Monthly Analysis of OCI", fontsize=16, weight='bold')
plt.xlabel("")
plt.ylabel("")
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0)

plt.tight_layout()
plt.show()



#6th objective:- To visualize total OCI-registration Recorded per year
data['Year'] = data['Date'].dt.year
oci_per_year = data.groupby('Year')["OCI - Registered"].sum()
print(oci_per_year)
plt.figure(figsize=(10, 6))
oci_per_year.plot(kind='bar', color='skyblue')
plt.title("Total OCI Registered Records Per Year")
plt.xlabel("Year")
plt.ylabel("Number of OCI Registrations")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


