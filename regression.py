# Do linear regressions. x: InsurTech index, y: Number of complaints.

import statsmodels.api as sm
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


# Load all input variables from 1 sheet
input_variables_path = os.environ.get("input_variables")
input_df = pd.read_excel(input_variables_path, "input")


