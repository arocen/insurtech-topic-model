# Do linear regressions. x: InsurTech index, y: Number of complaints.

import statsmodels.api as sm
import pandas as pd
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Load x (InsurTech index)
InsurTech_path = os.environ.get("report_insurtech_index")
index_df = pd.read_excel(InsurTech_path, "rescale", index_col=0)

# Load y (Number of complaints)
complaints_path = os.environ.get("complaints")
complaints_df = pd.read_excel(complaints_path, "总投诉量", index_col=0)

# Take the logarithm
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
for c in [c for c in complaints_df.columns if complaints_df[c].dtype in numerics]:
    complaints_df[c] = np.log10(complaints_df[c])

# Load premium income
income_path = os.environ.get("premium_income")
income_df = pd.read_excel(income_path, "原保费收入", index_col=0)

# Loop and do regressions
for company in list(index_df.index):
    X = index_df.loc[company] + income_df.loc[company]
    print(X)
    # X = sm.add_constant(X)
    # model = sm.OLS(complaints_df.loc[company], X).fit()
    # print(model.summary())