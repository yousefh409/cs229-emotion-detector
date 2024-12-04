import pytz
import os
import pandas as pd
import numpy as np

DEFAULT_TZ = pytz.FixedOffset(540)  # GMT+09:00; Asia/Seoul

PATH_DATA = 'data/SubjData/'
PATH_ESM = os.path.join(PATH_DATA, 'EsmResponse.csv')
PATH_PARTICIPANT = os.path.join(PATH_DATA, 'UserInfo.csv')

PATH_INTERMEDIATE = './intermediate'

SECOND_MS = 1000
MINUTE_MS = 60*SECOND_MS
DATA_TYPES = {
    'Acceleration': 'ACC',
    'AmbientLight': 'AML',
    'Calorie': 'CAL',
    'Distance': 'DST',
    'EDA': 'EDA',
    'HR': 'HRT',
    'RRI': 'RRI',
    'SkinTemperature': 'SKT',
    'StepCount': 'STP',
    'UltraViolet': 'ULV',
    'ActivityEvent': 'ACE',
    'ActivityTransition': 'ACT',
    'AppUsageEvent': 'APP',
    'BatteryEvent': 'BAT',
    'CallEvent': 'CAE',
    'Connectivity': 'CON',
    'DataTraffic': 'DAT',
    'InstalledApp': 'INS',
    'Location': 'LOC',
    'MediaEvent': 'MED',
    'MessageEvent': 'MSG',
    'WiFi': 'WIF',
    'ScreenEvent': 'SCR',
    'RingerModeEvent': 'RNG',
    'ChargeEvent': 'CHG',
    'PowerSaveEvent': 'PWS',
    'OnOffEvent': 'ONF'
}

def remove_mul_deltas(df):
  for column in df.columns:
    if column.count("-") > 1:
        if column in df.columns:
          df = df.drop(column, axis=1)
  return df


def get_prepared_data():
    pcodes = [f"P{str(i).zfill(2)}" for i in range(81)]
    all_data_df = pd.DataFrame()

    for pcode in pcodes:
        user_df = pd.DataFrame()

        for datatype in ["HR", "SkinTemperature", "Acceleration", "AmbientLight"]: # only uses these two sensor datas for now
            try:
                df = pd.read_csv(f"data/{pcode}/{datatype}.csv")
            except FileNotFoundError:
                continue
    
            df['pcode'] = pcode

            df["timestamp-1min"] = df["timestamp"] - MINUTE_MS
            df = pd.merge_asof(df, df[df.columns.difference(['pcode', "timestamp-1min"])], left_on="timestamp-1min", right_on="timestamp", suffixes=["", "-1min"], direction="nearest", tolerance=1500)
            df = remove_mul_deltas(df)
            df = df.drop("timestamp-1min", axis=1)

            df["timestamp-5min"] = df["timestamp"] - 5*MINUTE_MS
            df = pd.merge_asof(df, df[df.columns.difference(['pcode', "timestamp-5min"])], left_on="timestamp-5min", right_on="timestamp", suffixes=["", "-5min"], direction="nearest", tolerance=1500)
            df = remove_mul_deltas(df)
            df = df.drop("timestamp-5min", axis=1)

            df["timestamp-10min"] = df["timestamp"] - 10*MINUTE_MS
            df = pd.merge_asof(df, df[df.columns.difference(['pcode', "timestamp-10min"])], left_on="timestamp-10min", right_on="timestamp", suffixes=["", "-10min"], direction="nearest", tolerance=1500)
            df = remove_mul_deltas(df)
            df = df.drop("timestamp-10min", axis=1)

            if user_df.empty:
                user_df = df
            else:
                user_df = pd.merge_asof(user_df, df, on=["timestamp"], by=["pcode"], direction="nearest", tolerance=1500)
                # user_df = pd.merge(
                #   user_df,
                #   df,
                #   how="inner",
                #   on=['pcode', 'timestamp'],
                # )
                user_df = user_df.dropna()

        # all_data_df.isnull().mean() * 100
        all_data_df = pd.concat([all_data_df, user_df])
        all_data_df = all_data_df.dropna()

    esm_response = pd.read_csv('data/SubjData/EsmResponse.csv')
    joined_df = pd.merge(
            all_data_df,
            esm_response,
            how="inner",
            left_on=['pcode'],
            right_on=['pcode']
            )
    
    threshold = 60000 # questionare completed within a minute of sensor readings
    joined_df = joined_df[abs(joined_df['timestamp'] - joined_df['responseTime']) <= threshold]
    df = joined_df.reset_index(drop=True)
    df = df.drop(columns=['timestamp', 'responseTime', 'scheduledTime', 'duration', 'disturbance', 'change'])

    df["accel"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2 + df["z"] ** 2)
    df["accel-1min"] = np.sqrt(df["x-1min"] ** 2 + df["y-1min"] ** 2 + df["z-1min"] ** 2)
    df["accel-5min"] = np.sqrt(df["x-5min"] ** 2 + df["y-5min"] ** 2 + df["z-5min"] ** 2)
    df["accel-10min"] = np.sqrt(df["x-10min"] ** 2 + df["y-10min"] ** 2 + df["z-10min"] ** 2)
    df = df.drop(columns=['x', 'y', 'z', 'x-1min', 'y-1min', 'z-1min', 'x-5min', 'y-5min', 'z-5min', 'x-10min', 'y-10min', 'z-10min'])

    df = df.sample(frac=1)

    return df