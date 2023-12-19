# Run regressions on panel data.

import pandas as pd
from linearmodels import PanelOLS
import os
from dotenv import load_dotenv

load_dotenv()


def runModel(input_df:pd.DataFrame, x_labels:list[str]):
    '''Run a fixed effects regression'''

    # Set index
    panel_data = input_df.set_index(['company', 'year'])

    # Run a fixed effects regression
    model = PanelOLS(panel_data['总投诉量'], panel_data[x_labels], entity_effects=True, time_effects=True)
    results = model.fit()

    # Display regression results
    print(results)

    return

# Load panel data from Excel
input_variables_path = os.environ.get("input_variables")
inputEst_df = pd.read_excel(input_variables_path, "input_est", skiprows=[1, 2, 3])
# print(inputEst_df)

# runModel(inputEst_df, ['保险科技指标'])
# runModel(inputEst_df, ['保险科技指标', '原保费收入'])
# runModel(inputEst_df, ['保险科技指标', '原保费收入', '赔付支出'])