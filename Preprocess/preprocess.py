from typing import Optional
import numpy as np
import pandas as pd


def resample(df: pd.DataFrame, resample_freq: str = '200us'):
    """
    resamples the data with frequency 5000hz = 1/5000sec = 1/200microsec
    :param df:
    :param resample_freq:
    :return:
    """
    df = df.resample(resample_freq).mean()
    return df


def interpolate(df: pd.DataFrame, limit: int = 20, method: str = 'linear', order: Optional[int] = None):
    df = df.interpolate(method=method, order=order, limit=limit)
    return df


if __name__ == '__main__':
    df = pd.DataFrame([(0.0, np.nan, -1.0, 1.0),
                       (np.nan, 2.0, np.nan, np.nan),
                       (2.0, 3.0, np.nan, 9.0),
                       (np.nan, 4.0, -4.0, 16.0)],
                      columns=list('abcd'))
    print(df)
    df = interpolate(df)
    print(df)
