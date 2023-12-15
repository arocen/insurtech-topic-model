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

# # 平安
# # Add a constant term for intercept
# X_pingan = sm.add_constant(index_df.loc['平安'])

# # Fit the linear regression model
# model = sm.OLS(complaints_df.loc['平安'], X_pingan).fit()

# # Print the regression results
# print(model.summary())

# for rowlabel in ["平安"“]
print(list(index_df.index))