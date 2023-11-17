# parse result excel file of K-L divergence
import pandas as pd


# to-do: linearly scale K-L divergence the interval [0, 1]

def loadKL(path:str)->pd.DataFrame:
    '''Load KL results from excel file.'''
    df = pd.read_excel(path)
    return df

def linearScale(kl_df:pd.DataFrame):
    '''Linearly scale K-L divergence the interval [0, 1]'''
    # To-do
    

    return
