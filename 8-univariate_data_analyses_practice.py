# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 15:17:16 2021

@author: Banks Family
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
import numpy as np
from scipy.stats import iqr

da = pd.read_csv("nhanes_2015_2016.csv")

# ---------- Question 1 ------------------

# Relabel the marital status variable DMDMARTL to have brief but informative 
# character labels. Then construct a frequency table of these values for all 
# people, then for women only, and for men only. Then construct these three 
# frequency tables using only people whose age is between 30 and 40.

# relabeling variables to have character labels
da['DMDMARTLx'] = da.DMDMARTL.replace({1: "Married", 2: "Widowed", 3: "Divorced", 4: "Separated", 5: "Never Married",
                                        6: "Living with Partner", 77: "Refused", 99: "Don't Know"})
da['RIAGENDRx'] = da.RIAGENDR.replace({1: "Male", 2: "Female"})

# check age interval - should be between 30 and 40 (not inclusive) for both 
# methods
print("AGE INTERVAL CHECK\n")

da["agegroup"]=pd.cut(da.RIDAGEYR, bins=[31, 40], right=False)
print("GROUPBY METHOD")
print(da.groupby(["agegroup"])["RIDAGEYR"].min())
print(da.groupby(["agegroup"])["RIDAGEYR"].max())
print("")

# print("WHERE METHOD")
# selections = (da.RIDAGEYR < 40) & (da.RIDAGEYR > 30)
# print(da.where(selections).RIDAGEYR.min())
# print(da.where(selections).RIDAGEYR.max())
# print("")

# print frequencies
print("EVERYONE")
print(da.DMDMARTLx.value_counts())
print("")

print("USING PD.CUT and GROUPBY\n")
print("GENDER ONLY")
print(da.groupby(["RIAGENDRx"])["DMDMARTLx"].value_counts())
print("")

print("AGE ONLY")
print(da.groupby(["agegroup"])["DMDMARTLx"].value_counts())
print("")

print("AGE AND GENDER")
print(da.groupby(["agegroup", "RIAGENDRx"])["DMDMARTLx"].value_counts())
print("")

print("USING WHERE\n")
# .where methods only work in Jupiter notebook

# # Marital status frequency table for men only
# selections = (da.RIAGENDRx == "Male")
# print("MALES ONLY")
# print(da.where(selections).DMDMARTLx.value_counts())
# print("")
# print("")

# Martital status frequency table for women only
# selections = (da.RIAGENDRx == "Female")
# print("FEMALES ONLY")
# print(da.where(selections).DMDMARTLx.value_counts())
# print("")

# Marital status frequency table for both men and women between 30 and 40
# selections = (da.RIDAGEYR < 40) & (da.RIDAGEYR > 30)
# print("EVERYONE BETWEEN 30 AND 40")
# print(da.where(selections).DMDMARTLx.value_counts())
# print("")

# Marital status frequency table for men between 30 and 40
# selections = (da.RIAGENDRx == "Male") & (da.RIDAGEYR < 40) & (da.RIDAGEYR > 30)
# print("MALES ONLY BETWEEN 30 AND 40")
# print(da.where(selections).DMDMARTLx.value_counts())
# print("")

# Marital status frequency table for women between 30 and 40
# selections = (da.RIAGENDRx == "Female") & (da.RIDAGEYR < 40) & (da.RIDAGEYR > 30)
# print("FEMALES ONLY BETWEEN 30 AND 40")
# print(da.where(selections).DMDMARTLx.value_counts())

# ---------- Question 2 ------------------

# Restricting to the female population, stratify the subjects into age bands 
# no wider than ten years, and construct the distribution of marital status 
# within each age band. Within each age band, present the distribution in 
# terms of proportions that must sum to 1.

# re-label variables to descriptive labels
da['DMDMARTLx'] = da.DMDMARTL.replace({1: "Married", 2: "Widowed", 3: "Divorced", 4: "Separated", 5: "Never Married",
                                        6: "Living with Partner", 77: "Refused", 99: "Don't Know"})
da['RIAGENDRx'] = da.RIAGENDR.replace({1: "Male", 2: "Female"})

# restrict data to female population
df = da.where(da.RIAGENDR == 2)
print('FEMALES ONLY')
print('')

# cut data into bands no wider than 10 years
df['agegrp'] = pd.cut(df.RIDAGEYR, [20, 30, 40, 50, 60, 70, 80])

# eliminate the missing values
dx = df.loc[~da.DMDMARTLx.isin(["Don't know", 'Missing']), :]

# group marital status by age group bucket
dx = dx.groupby(['agegrp'])['DMDMARTLx']

# obtain the counts for marital status within each age group band
dx = dx.value_counts()
dx = dx.unstack() # Restructure the results from 'long' to 'wide'
dx = dx.apply(lambda x: x/x.sum(), axis = 1) # Normalize within each stratum to get proportions
print(dx.to_string(float_format = '%.3f')) # limit display to 3 decimal places

# # Repeat the construction for males.

# restrict data to male population
da = da.where(da.RIAGENDRx == 'Male')
print('MALES ONLY')
print('')

# cut data into bands no wider than 10 years
da['agegrp'] = pd.cut(da.RIDAGEYR, [20, 30, 40, 50, 60, 70, 80])

# eliminate the missing values
dx = da.loc[~da.DMDMARTLx.isin(["Don't know", 'Missing']), :]

# group marital status by age group bucket
dx = dx.groupby(['agegrp'])['DMDMARTLx']

# obtain the counts for marital status within each age group band
dx = dx.value_counts()

dx = dx.unstack() # Restructure the results from 'long' to 'wide'
dx = dx.apply(lambda x: x/x.sum(), axis = 1) # Normalize within each stratum to get proportions
print(dx.to_string(float_format = '%.3f')) # limit display to 3 decimal places


# ---------- Question 3 ------------------

# Construct a histogram of the distribution of heights using the BMXHT 
# variable in the NHANES sample.
sns.distplot(da.BMXHT.dropna())

# Use the 'bins' argument to distplot to produce histograms with different 
# numbers of bins. For example, give me 10 bins.
sns.distplot(da.BMXHT.dropna(), bins=10)

# # Make separate histograms for the heights of women and men, then make a 
# # side-by-side boxplot showing the heights of women and men.

# re-label variables to descriptive labels
da["RIAGENDRx"] = da.RIAGENDR.replace({1: "Male", 2: "Female"})

# specify data to include only females and males, respectively
df = da.loc[da.RIAGENDRx.isin(["Female"]), :]
dm = da.loc[da.RIAGENDRx.isin(["Male"]), :]

# create histogram for females only
sns.distplot(df.BMXHT.dropna())

# create histogram for males only
sns.distplot(dm.BMXHT.dropna())

# Side-by-side boxplots for women and men
sns.boxplot(x="RIAGENDRx", y="BMXHT", data=da)

# ---------- Question 4 ------------------
# Make a boxplot showing the distribution of within-subject differences 
# between the first and second systolic blood pressure measurents 
# (BPXSY1 and BPXSY2). In other words, what proportion of the BPXSY1 
# measurements are larger than the corresponding BPXSY2 measurements?


# Not sure what the question is looking for; thought that they were looking
# for the below boxplot.

# Make side-by-side boxplots of the two systolic blood pressure variables.
sns.boxplot(data = da.loc[:, ['BPXSY1', 'BPXSY2']])

# # ---------- Question 5 ------------------

# Construct a frequency table of household sizes for people within each 
# educational attainment category (the relevant variable is DMDEDUC2). Convert 
# the frequencies to proportions.

da['DMDEDUC2x'] = da.DMDEDUC2.replace({1: '<9', 2: '9-11', 3: 'HS/GED', 4:
                                        'Some College/AA', 5: 'College', 7:
                                        'Refused', 9: "Didn't know"})
dx = da.groupby('DMDEDUC2x')['DMDHHSIZ'].value_counts()
dx = dx.unstack()
dx = dx.apply(lambda x: x/x.sum(), axis = 1)
print(dx.to_string(float_format = '%.3f'))

# Restrict the sample to people between 30 and 40 years of age. Then calculate 
# the median household size for women and men within each level of educational 
# attainment.
da['agegrp_30_40'] = pd.cut(da.RIDAGEYR, [30, 40])
da = da.groupby(['agegrp_30_40', 'RIAGENDRx', 'DMDEDUC2x'])['DMDHHSIZ']
print(da.median())

# # ---------- Question 6 ------------------

# The participants can be clustered into "maked variance units" (MVU) based 
# on every combination of the variables SDMVSTRA and SDMVPSU. Calculate the 
# mean age (RIDAGEYR), height (BMXHT), and BMI (BMXBMI) for each 
# gender (RIAGENDR), within each MVU, and report the ratio between the 
# largest and smallest mean (e.g. for height) across the MVUs.

# create new dataframes for males and females
dam = da.where(da.RIAGENDR == 1)
daf = da.where(da.RIAGENDR == 2)

# -------------------------------------
## Mean age for each MVU and the ratio between max/min mean across MVUs
# calculate max means for age across each combo of SDMVPSU and SDMVSTRA
agemax_male = dam.groupby(['SDMVPSU', 'SDMVSTRA'])["RIDAGEYR"].mean().max()
agemax_female = daf.groupby(['SDMVPSU', 'SDMVSTRA'])["RIDAGEYR"].mean().max()

# # calculate min means for age across each combo of SDMVPSU and SDMVSTRA
agemin_male = daf.groupby(['SDMVPSU', 'SDMVSTRA'])["RIDAGEYR"].mean().min()
agemin_female = daf.groupby(['SDMVPSU', 'SDMVSTRA'])["RIDAGEYR"].mean().min()

# print max, mean and ratio for males
print("AGES")
print("MALES:")
print("Male agemax:", agemax_male)
print("Male agemin:", agemin_male)
print("Male age ratio:", agemax_male/agemin_male)
print("")

# print max, mean and ratio for males
print("FEMALES:")
print("Female agemax:", agemax_female)
print("Female agemin:", agemin_female)
print("Female age ratio:", agemax_female/agemin_female)

### Mean height (BMXHT) for each MVU and the ratio between max/min mean across MVUs
# calculate max means for height across each combo of SDMVPSU and SDMVSTRA
heightmax_male = dam.groupby(['SDMVPSU', 'SDMVSTRA'])["BMXHT"].mean().max()
heightmax_female = daf.groupby(['SDMVPSU', 'SDMVSTRA'])["BMXHT"].mean().max()

# calculate min means for height across each combo of SDMVPSU and SDMVSTRA
heightmin_male = daf.groupby(['SDMVPSU', 'SDMVSTRA'])["BMXHT"].mean().min()
heightmin_female = daf.groupby(['SDMVPSU', 'SDMVSTRA'])["BMXHT"].mean().min()

# print max, mean and ratio for males
print("-----------------")
print("HEIGHTS")
print("MALES:")
print("Male height max:", heightmax_male)
print("Male height min:", heightmin_male)
print("Male height ratio:", heightmax_male/heightmin_male)
print("")

# print max, mean and ratio for males
print("FEMALES:")
print("Female height max:", heightmax_female)
print("Female height min:", heightmin_female)
print("Female height ratio:", heightmax_female/heightmin_female)

### Mean BMI (BMXBMI) for each MVU and the ratio between max/min mean across MVUs
# calculate max means for height across each combo of SDMVPSU and SDMVSTRA
bmimax_male = dam.groupby(['SDMVPSU', 'SDMVSTRA'])["BMXBMI"].mean().max()
bmimax_female = daf.groupby(['SDMVPSU', 'SDMVSTRA'])["BMXBMI"].mean().max()

bmimin_male = daf.groupby(['SDMVPSU', 'SDMVSTRA'])["BMXBMI"].mean().min()
bmimin_female = daf.groupby(['SDMVPSU', 'SDMVSTRA'])["BMXBMI"].mean().min()

# print max, mean and ratio for males
print("-----------------")
print("BMI")
print("MALES:")
print("Male BMI max:", bmimax_male)
print("Male BMI min:", bmimin_male)
print("Male BMI ratio:", bmimax_male/bmimin_male)
print("")

# print max, mean and ratio for males
print("FEMALES:")
print("Female BMI max:", bmimax_female)
print("Female BMI min:", bmimin_female)
print("Female BMI ratio:", bmimax_female/bmimin_female)
print("-----------------")

### Calculate the inter-quartile range (IQR) for age, height, and BMI for each 
# gender and each MVU. Report the ratio between the largest and smalles IQR 
# across the MVUs.

# drop the missing data
print("\nAGGREGATE IQR")
print("RIDAGEYR IQR: ", iqr(da.RIDAGEYR.dropna()))
print("BMXHT IQR: ", iqr(da.BMXHT.dropna()))
print("BMXBMI IQR: ", iqr(da.BMXBMI.dropna()))

# calculate IQRs for age, height, and BMI for men
print("\nMALE IQR")
print("RIDAGEYR: ", iqr(dam.RIDAGEYR.dropna()[dam.RIAGENDR == 1]))
print("BMXHT: ", iqr(dam.BMXHT.dropna()[dam.RIAGENDR == 1]))
print("BMXBMI: ", iqr(dam.BMXBMI.dropna()[dam.RIAGENDR == 1]))

# calculate the IQRs for age, height, and BMI for women
print("\nFEMALE IQR")
print("RIDAGEYR: ", iqr(daf.RIDAGEYR.dropna()[daf.RIAGENDR == 2]))
print("BMXHT: ", iqr(daf.BMXHT.dropna()[daf.RIAGENDR == 2]))
print("BMXBMI: ", iqr(daf.BMXBMI.dropna()[daf.RIAGENDR == 2]))

# calculate the IQR by age for each MVU 
RIDAGEYRGrp = da[da.RIDAGEYR.notna()].groupby(['SDMVSTRA', 'SDMVPSU'])
print("\nRIDAGEYR GROUPED")
print(RIDAGEYRGrp["RIDAGEYR"].agg(iqr))

# calculate the IQR by height for each MVU 
BMXHTGrp = da[da.BMXHT.notna()].groupby(['SDMVSTRA', 'SDMVPSU'])
print("\nBMXHT GROUPED")
print(BMXHTGrp["BMXHT"].agg(iqr))

# calculate the IQR by BMI for each MVU
BMXBMIGrp = da[da.BMXBMI.notna()].groupby(['SDMVSTRA', 'SDMVPSU'])
print("\nBMXBMI GROUPED")
print(BMXBMIGrp["BMXBMI"].agg(iqr))

# calculate the ratio between largest and smallest age IQR across MVUs
print("\nRatio of Max/Min for RIDAGEYR")
print(max(RIDAGEYRGrp["RIDAGEYR"].agg(iqr)) / min(RIDAGEYRGrp["RIDAGEYR"].agg(iqr)))

# calculate the ratio between largest and smallest height IQR across MVUs
print("\nRatio of Max/Min for BMXBHT")
print(max(BMXHTGrp["BMXHT"].agg(iqr)) / min(BMXHTGrp["BMXHT"].agg(iqr)))

# calculate the ratio between largest and smallest BMI IQR across MVUs
print("\nRatio of Max/Min for BMXBMI")
print(max(BMXBMIGrp["BMXBMI"].agg(iqr)) / min(BMXBMIGrp["BMXBMI"].agg(iqr)))





