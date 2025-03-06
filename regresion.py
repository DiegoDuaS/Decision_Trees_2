import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.stats.diagnostic as diag
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder

def trans_categorical(df):
    df = df.copy()  # Evitar modificar el dataframe original
    
    # Eliminar las variables que no queremos en el análisis de clusters
    drop_columns = [
        'Id', 'PoolArea', 'MiscVal', 'BsmtFinSF2', 'BsmtFinSF1', 'MasVnrArea',
        'BsmtUnfSF', '2ndFlrSF', 'LowQualFinSF', 'WoodDeckSF', 'OpenPorchSF',
        'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'Alley', 'ExterCond',
        'BsmtHalfBath', 'KitchenAbvGr', 'PoolQC', 'Fence', 'MiscFeature', 'MiscFeature',
        'FireplaceQu', 'MasVnrType', 
    ]
    df = df.drop(columns=drop_columns, errors='ignore')
    
    return df
