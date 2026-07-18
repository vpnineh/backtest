import os
import glob
import zipfile
import pandas as pd
import numpy as np
import json
from datetime import datetime


INITIAL_BALANCE = 10000
RISK_A = 0.005
RISK_B = 0.01
RISK_C = 0.015


def load_files():

    files = glob.glob("data/*")

    csv_files = []

    for f in files:

        if f.endswith(".zip"):
            with zipfile.ZipFile(f) as z:
                for name in z.namelist():
                    if name.endswith(".csv"):
                        z.extract(name,"data/temp")
                        csv_files.append(
                            "data/temp/" + name
                        )

        elif f.endswith(".csv"):
            csv_files.append(f)


    return csv_files



def read_data(file):

    try:

        df=pd.read_csv(
            file,
            sep=";",
            header=None
        )

        if len(df.columns)>=6:

            df=df.iloc[:,:6]

            df.columns=[
                "time",
                "open",
                "high",
                "low",
                "close",
                "volume"
            ]


        else:

            return None


        df["time"]=pd.to_datetime(
            df["time"],
            errors="coerce"
        )

        df=df.dropna()

        for c in [
            "open",
            "high",
            "low",
            "close"
        ]:
            df[c]=pd.to_numeric(
                df[c],
                errors="coerce"
            )


        return df.sort_values("time")


    except Exception:

        return None



def indicators(df):

    df["ema50"]=(
        df.close
        .ewm(span=50)
        .mean()
    )

    df["ema200"]=(
        df.close
        .ewm(span=200)
        .mean()
    )


    delta=df.close.diff()

    gain=delta.clip(lower=0)
    loss=-delta.clip(upper=0)


    rs=(
        gain.rolling(14).mean()
        /
        loss.rolling(14).mean()
    )

    df["rsi"]=100-(100/(1+rs))


    tr=pd.concat(
        [
            df.high-df.low,
            abs(df.high-df.close.shift()),
            abs(df.low-df.close.shift())
        ],
        axis=1
    ).max(axis=1)


    df["atr"]=tr.rolling(14).mean()


    return df.dropna()



def backtest(df,mode):

    balance=INITIAL_BALANCE
    equity=[]

    trades=[]

    position=None


    for i,row in df.iterrows():


        price=row.close


        signal=None


        # MODE A
        if mode=="A":

            if (
                row.ema50>row.ema200
                and row.rsi>45
                and row.rsi<65
            ):
                signal="BUY"


            elif (
                row.ema50<row.ema200
                and row.rsi>35
                and row.rsi<55
            ):
                signal="SELL"



        # MODE B
        elif mode=="B":

            if row.rsi<30:

                signal="BUY"

            elif row.rsi>70:

                signal="SELL"



        # MODE C

        elif mode=="C":

            if (
                price>row.ema50
                and row.atr>df.atr.mean()
            ):
                signal="BUY"

            elif (
                price<row.ema50
                and row.atr>df.atr.mean()
            ):
                signal="SELL"



        if signal:

            risk={
                "A":RISK_A,
                "B":RISK_B,
                "C":RISK_C
            }[mode]


            sl=row.atr*1.5
            tp=row.atr*3


            if signal=="BUY":

                result=(
                    df.close.iloc[
                        min(i+20,len(df)-1)
                    ]
                    -
                    price
                )


            else:

                result=(
                    price-
                    df.close.iloc[
                        min(i+20,len(df)-1)
                    ]
                )


            money=balance*risk


            if result>tp:

                balance+=money*2

                trades.append(1)


            elif result<-sl:

                balance-=money

                trades.append(-1)



        equity.append(balance)



    wins=len(
        [x for x in trades if x>0]
    )

    losses=len(
        [x for x in trades if x<0]
    )


    dd=(
        1-
        min(equity)/
        max(equity)
    )*100


    return {

        "mode":mode,
        "balance":round(balance,2),
        "profit_%":
            round(
                (balance/INITIAL_BALANCE-1)*100,
                2
            ),

        "trades":len(trades),

        "winrate":
            round(
                wins/max(1,len(trades))*100,
                2
            ),

        "drawdown_%":
            round(dd,2)

    }



def main():

    files=load_files()

    results=[]


    for f in files:

        print(
            "Testing:",
            f
        )

        df=read_data(f)

        if df is None:
            continue


        df=df.resample(
            "1H",
            on="time"
        ).agg(
            {
                "open":"first",
                "high":"max",
                "low":"min",
                "close":"last",
                "volume":"sum"
            }
        ).dropna()


        df=indicators(df)


        for mode in [
            "A",
            "B",
            "C"
        ]:

            r=backtest(
                df,
                mode
            )

            r["file"]=os.path.basename(f)

            results.append(r)



    os.makedirs(
        "results",
        exist_ok=True
    )


    with open(
        "results/report.json",
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )


    pd.DataFrame(results).to_csv(
        "results/report.csv",
        index=False
    )


    print("\nDONE")
    print(
        json.dumps(
            results,
            indent=2
        )
    )



if __name__=="__main__":
    main()
