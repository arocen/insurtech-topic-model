# Do linear regressions. x: InsurTech index, y: Number of complaints.

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

def regression1(input_df):
    '''Assume: number of complaints = β0 + β1 * InsurTech index + β2 * Premium income'''
    # Do regression for all companies first
    X = input_df[["保险科技指标", "原保费收入"]]
    X = sm.add_constant(X)
    y = input_df["总投诉量"]

    model = sm.OLS(y, X).fit()
    print(model.summary())
    return

def regressionAvg(input_df):
    '''Assume: number of complaints / Premium income = β0 + β1 * InsurTech index'''
    X = input_df["保险科技指标"]
    X = sm.add_constant(X)
    y = input_df["亿元保费投诉量"]

    model = sm.OLS(y, X).fit()
    print(model.summary())
    return

def regressionRm(input_df:pd.DataFrame):
    '''
    Remove rows of 人保 in 2015, 2016, 2017.
    Assume: number of complaints = β0 + β1 * InsurTech index + β2 * Premium income
    '''
    input_df = input_df.drop([0, 1, 2])
    print(input_df)

    X = input_df[["保险科技指标", "原保费收入"]]
    checkVif(X)

    X = sm.add_constant(X)
    y = input_df["总投诉量"]

    model = sm.OLS(y, X).fit()
    print(model.summary())
    return

def checkVif(X):
    '''Use variance inflation factor (VIF) values to quantify the severity of multicollinearity. High VIF values (typically above 10) may indicate a problem.'''
    # Assuming X is your design matrix (independent variables)
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print(vif_data)
    return



# Load all input variables from 1 sheet
input_variables_path = os.environ.get("input_variables")
input_df = pd.read_excel(input_variables_path, "input")


# regressionAvg(input_df)
regressionRm(input_df)
