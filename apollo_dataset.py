from torch.utils.data import Dataset
import pandas as pd
from sklearn import preprocessing
from pandas.api.types import is_string_dtype, is_numeric_dtype
import math
import random
import torch

class ApolloDataset(Dataset):
    def __init__(self, is_train):
        self.TRAIN_PORTION = 0.7
        self.NAN_TOLERANCE = 0.5
        self.is_train = is_train
        self.file_location = "kidney_disease.csv"
        csv_data = pd.read_csv("kidney_disease.csv")
        self.total = len(csv_data)

        df_normalized = pd.DataFrame(csv_data)
        df_normalized = self._omit_empty_columns(df_normalized)
        df_normalized = self._normalize(df_normalized)
        self.rows = [row for index, row in df_normalized.iterrows()]
        self.train_count = int(self.total * self.TRAIN_PORTION)
        self.count = self.train_count
        self.start_index = 0

        if self.is_train is False:
            self.count = len(self.rows) - self.train_count
            self.start_index = self.train_count

        self.dim = 0
        for index, row in df_normalized.iterrows():
            if index > 0:
                for cell in row:
                    if torch.is_tensor(cell):
                        self.dim = self.dim + cell.shape[0]
                        print(cell.shape)
                    else:
                        print(cell)
                        self.dim = self.dim + 1
                break
        print(self.dim)


    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        return 1,1

    def _normalize(self, df):
        for col in df.columns:
            if is_numeric_dtype(df[col]):
                df = self._normalize_numeric(df, col)
            elif is_string_dtype(df[col]):
                df = self._normalize_string(df, col)
        return df

    def _normalize_string(self,df, col):
        self._fill_empty_string(df, col)
        uniques = list(df[col].unique())
        for i in range(len(df[col])):
            str = df[col][i]
            value = torch.zeros(len(uniques))
            value[uniques.index(str)] = 1
            df.at[i,col] = value
        return df

    def _fill_empty_string(self, df, col):
        count_nans = df[col].isna().sum()
        if count_nans == 0:
            return df
        dist = []
        for cell in df[col]:
            if cell == cell:
                dist.append(cell)

        for i in range(len(df[col])):
            cell = df[col][i]
            if cell != cell:
                df.at[i, col] = "unknown"
        return df

    def _normalize_numeric(self,df, col):
        x = df[[col]].values.astype(float)
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df_normalized = df
        df_normalized[col] = x_scaled
        mean = df[col].mean()
        df.fillna({col:mean}, inplace=True)
        return df

    def _omit_empty_columns(self, df):
        omit_list = []
        for col in df.columns:
            count_nan = df[col].isna().sum()
            ratio = count_nan / self.total
            if ratio > self.NAN_TOLERANCE:
                omit_list.append(col)
        return df.drop(columns=omit_list)

if __name__ == "__main__":
    d = ApolloDataset(is_train=True)

