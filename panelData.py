# Run regressions on panel data.

import pandas as pd
from linearmodels import PanelOLS
import os
from dotenv import load_dotenv

load_dotenv()


def runModel(input_df:pd.DataFrame, x_labels:list[str], y_label='总投诉量', year_fe=True, company_fe=True, cov_type='unadjusted'):
    '''
    Run a fixed effects regression.
    By default, linearmodels includes an intercept term in the regression unless you explicitly remove it.
    
    - cov_type: "unadjusted" means assuming residual are homoskedastic.
        "robust" means control for heteroskedasticity using White's estimator.
        "kenel" means Bartlett's kernel, which is produces a covariance estimator similar to the Newey-West covariance estimator.
        This does not infulence the results of regression, but just F-statistic (robust).
    '''

    # Set index
    panel_data = input_df.set_index(['company', 'year'])

    # Run a fixed effects regression
    model = PanelOLS(panel_data[y_label], panel_data[x_labels], time_effects=year_fe, entity_effects=company_fe)
    results = model.fit(cov_type=cov_type)

    # Display regression results
    print(results)

    return


# Load panel data from Excel
input_variables_path = os.environ.get("input_variables")
inputEst_df = pd.read_excel(input_variables_path, "input_est", skiprows=[1, 2, 3])
# print(inputEst_df)

# runModel(inputEst_df, ['保险科技指标'])
# runModel(inputEst_df, ['保险科技指标', '原保费收入'])
# runModel(inputEst_df, ['保险科技指标', '赔付率'])
# runModel(inputEst_df, ['保险科技指标', '赔付率'], y_label="log_总投诉量")
# runModel(inputEst_df, ['保险科技指标', '原保费收入', '赔付支出'], cov_type='robust')
# runModel(inputEst_df, ['保险科技指标_refer3', '原保费收入', '赔付支出'], cov_type='robust', year_fe=False)
# runModel(inputEst_df, ['保险科技指标', '原保费收入', '赔付支出'], cov_type='robust', year_fe=False)
# runModel(inputEst_df, ['保险科技指标', '原保费收入', '赔付支出'], cov_type='robust', year_fe=True)