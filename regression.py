# Do linear regressions. x: InsurTech index, y: Number of complaints.

import statsmodels.api as sm
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

# Load x (InsurTech index)
InsurTech_path = os.environ.get("report_insurtech_index")
index_df = pd.read_excel(InsurTech_path, "rescale", index_col=0)

# Load y (Number of complaints)
complaints_path = os.environ.get("complaints")
complaints_df = pd.read_excel(complaints_path, "总投诉量", index_col=0)

# Loop and do regressions
for company in list(index_df.index):
    X = sm.add_constant(index_df.loc[company])
    model = sm.OLS(complaints_df.loc[company], X).fit()
    print(model.summary())