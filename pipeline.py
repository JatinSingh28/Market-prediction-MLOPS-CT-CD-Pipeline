import pandas as pd
import numpy as np
from zenml import step, pipeline
import yfinance as yf
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from typing import Tuple
from typing_extensions import Annotated
import mlflow


@step
def importer(startDate: str, symbol: str) -> np.ndarray:
    endDate = datetime.today().strftime("%Y-%m-%d")
    data = yf.download(symbol, start=startDate, end=endDate)
    data = pd.DataFrame(data)["Close"].values
    data = np.array(data)
    # print(type(data))
    return data


@step(enable_cache=True)
def create_seq(data: np.ndarray, seq_len: int) -> Annotated[list, "Sequence"]:
    seq = []
    # print(len(data))
    for i in range(len(data) - seq_len):
        # print(i)
        tseq = data[i : (i + seq_len)]
        # print(tseq)
        label = data[i + seq_len]
        tseq = np.array(tseq)
        tseq = tseq.reshape(1, -1)[0]
        seq.append([tseq, label])
        # print(i)
    # print(seq)
    return seq


@step
def xy_train(
    sequence: list,
) -> Tuple[Annotated[np.ndarray, "X_train"], Annotated[np.ndarray, "Y_train"]]:
    x_train = np.array([x[0] for x in sequence])
    y_train = np.array([x[1] for x in sequence])
    print(x_train)
    return x_train, y_train


@step
def train(x_train: np.ndarray, y_train: np.ndarray) -> Annotated[Sequential,"Model"]:
    model = Sequential()
    model.add(
        LSTM(
            50,
            activation="relu",
            input_shape=(x_train.shape[1], 1),
            return_sequences=True,
        )
    )
    model.add(LSTM(100, activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    model.fit(x_train, y_train, epochs=25, batch_size=64, verbose=1)
    return model


@pipeline
def trainPipeline(seqLen: int, startDate: str = "2010-01-01", symbol: str = "^NSEI"):
    data = importer(startDate, symbol)
    seq = create_seq(data, seqLen)
    x_train, y_train = xy_train(seq)
    model = train(x_train,y_train)


if __name__ == "__main__":
    mlflow.set_tracking_uri("s3://your-s3-bucket-name/path/to/experiments")
    trainPipeline(startDate="2023-01-01", seqLen=20)
