import os
import glob
import zipfile
import json
import shutil
import pandas as pd
import numpy as np


# =========================
# CONFIG
# =========================

INITIAL_BALANCE = 10000

RISK = {
    "A": 0.005,
    "B": 0.01,
    "C": 0.015
}

ATR_MULT_SL = 1.5
ATR_MULT_TP = 3

RESULT_DIR = "results"
TEMP_DIR = "temp_data"


# =========================
# FILE LOADER
# =========================


def collect_files():

    files = []

    os.makedirs(TEMP_DIR, exist_ok=True)


    for f in glob.glob("data/*"):

        if f.endswith(".csv"):
            files.append(f)


        elif f.endswith(".zip"):

            with zipfile.ZipFile(f) as z:

                for name in z.namelist():

                    if name.endswith(".csv"):

                        out = os.path.join(
                            TEMP_DIR,
                            os.path.basename(name)
                        )

                        with open(out,"wb") as w:
                            w.write(
                                z.read(name)
                            )

                        files.append(out)


    return files



# =========================
# CSV PARSER
# =========================


def load_csv(path):

    try:

        df = pd.read_csv(
            path,
            sep=None,
            engine="python",
            header=None
        )


        # remove empty columns

        df = df.dropna(
            axis=1,
            how="all"
        )


        if len(df.columns) < 5:
            return None



        # HISTDATA FORMAT

        if len(df.columns) >= 6:


            df = df.iloc[:,0:6]

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



        df["time"] = pd.to_datetime(
            df["time"],
            errors="coerce"
        )


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


        df=df.dropna()


        df=df.sort_values(
            "time"
        )


        return df


    except Exception as e:

        print(
            "LOAD ERROR:",
            path,
            e
        )

        return None



# =========================
# INDICATORS
# =========================


def add_indicators(df):


    df["ema50"] = (
        df.close
        .ewm(span=50)
        .mean()
    )


    df["ema200"] = (
        df.close
        .ewm(span=200)
        .mean()
    )



    delta=df.close.diff()


    gain=np.where(
        delta>0,
        delta,
        0
    )


    loss=np.where(
        delta<0,
        -delta,
        0
    )


    avg_gain=pd.Series(
        gain
    ).rolling(14).mean()


    avg_loss=pd.Series(
        loss
    ).rolling(14).mean()


    rs=avg_gain / avg_loss


    df["rsi"] = (
        100 -
        (100/(1+rs))
    )



    tr=pd.concat(
        [
            df.high-df.low,
            abs(df.high-df.close.shift()),
            abs(df.low-df.close.shift())
        ],
        axis=1
    ).max(axis=1)



    df["atr"] = (
        tr.rolling(14)
        .mean()
    )


    df["bb_mid"]=(
        df.close
        .rolling(20)
        .mean()
    )


    std=(
        df.close
        .rolling(20)
        .std()
    )


    df["bb_high"]=df.bb_mid+2*std
    df["bb_low"]=df.bb_mid-2*std



    return df.dropna()



# =========================
# STRATEGY
# =========================


def get_signal(row,mode):


    price=row.close



    if mode=="A":

        if (
            row.ema50 >
            row.ema200
            and
            45 < row.rsi < 65
        ):
            return "BUY"



        if (
            row.ema50 <
            row.ema200
            and
            35 < row.rsi < 55
        ):
            return "SELL"



    elif mode=="B":


        if price < row.bb_low and row.rsi < 35:
            return "BUY"



        if price > row.bb_high and row.rsi > 65:
            return "SELL"



    elif mode=="C":


        if (
            price > row.ema50
            and
            row.atr >
            row.atr.mean()
        ):
            return "BUY"



        if (
            price < row.ema50
            and
            row.atr >
            row.atr.mean()
        ):
            return "SELL"



    return None




# =========================
# BACKTEST
# =========================


def run_backtest(df,mode):


    balance=INITIAL_BALANCE


    equity=[]

    trades=[]


    i=0


    while i < len(df)-10:


        row=df.iloc[i]


        signal=get_signal(
            row,
            mode
        )


        if signal:


            entry=row.close


            atr=row.atr


            sl_distance=atr*ATR_MULT_SL

            tp_distance=atr*ATR_MULT_TP



            result=0



            future=df.iloc[
                i+1:
                min(
                    i+50,
                    len(df)
                )
            ]



            for candle in future.itertuples():


                if signal=="BUY":


                    if candle.low <= entry-sl_distance:

                        result=-1
                        break


                    if candle.high >= entry+tp_distance:

                        result=2
                        break



                else:


                    if candle.high >= entry+sl_distance:

                        result=-1
                        break


                    if candle.low <= entry-tp_distance:

                        result=2
                        break



            money=(
                balance*
                RISK[mode]
            )


            if result==2:

                balance += money*2
                trades.append(1)



            elif result==-1:

                balance -= money
                trades.append(-1)



        equity.append(balance)

        i+=1



    wins=sum(
        x>0 for x in trades
    )


    losses=sum(
        x<0 for x in trades
    )


    peak=max(equity) if equity else INITIAL_BALANCE


    dd=[]


    for x in equity:

        if x>peak:
            peak=x

        dd.append(
            (peak-x)/peak*100
        )


    return {

        "mode":mode,

        "final_balance":
        round(balance,2),

        "profit_percent":
        round(
            (balance/INITIAL_BALANCE-1)*100,
            2
        ),

        "trades":
        len(trades),

        "win_rate":
        round(
            wins/max(1,len(trades))*100,
            2
        ),

        "max_drawdown":
        round(
            max(dd) if dd else 0,
            2
        )

    }




# =========================
# MAIN
# =========================


def main():


    if os.path.exists(RESULT_DIR):
        shutil.rmtree(
            RESULT_DIR
        )


    os.makedirs(
        RESULT_DIR
    )



    results=[]


    files=collect_files()


    print(
        "FILES:",
        len(files)
    )



    for file in files:


        print(
            "Testing:",
            file
        )


        df=load_csv(file)


        if df is None:
            continue



        df=df.set_index(
            "time"
        )



        df=df.resample(
            "1h"
        ).agg(
            {
                "open":"first",
                "high":"max",
                "low":"min",
                "close":"last",
                "volume":"sum"
            }
        ).dropna()



        if len(df)<300:
            continue



        df=add_indicators(df)



        for mode in [
            "A",
            "B",
            "C"
        ]:


            r=run_backtest(
                df,
                mode
            )


            r["symbol_file"]=(
                os.path.basename(file)
            )


            results.append(r)




    with open(
        RESULT_DIR+"/report.json",
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )



    pd.DataFrame(
        results
    ).to_csv(
        RESULT_DIR+"/report.csv",
        index=False
    )



    print("\n========== DONE ==========")


    for r in results:

        print(r)



if __name__=="__main__":
    main()
