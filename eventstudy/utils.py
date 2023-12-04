import sys
import traceback
import datetime
from deltalake import DeltaTable
from deltalake.writer import write_deltalake
import numpy as np
import pandas as pd
from scipy.stats import t
import pgdb2
import boto3
import pandas_datareader as pdr
import warnings
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from .exception import FutureDateError

warnings.simplefilter(action='ignore', category=Warning)



# All model must returns : (residuals: list, df: int, var: float)

# TODO: sortir la computation des résiduals des fonctions de modélisation. Juste leur faire calculer les prédictions. Sortir aussi windowsize estimation size et tout le reste, et aussi le secReturns qui doit être rataché à l'event study pas la fonction de modélisation.

def to_table(columns, asterisks_dict=None, decimals=None, index_start=0):

    if decimals:
        if type(decimals) is int:
            decimals = [decimals] * len(columns)

        for key, decimal in zip(columns.keys(), decimals):
            if decimal:
                columns[key] = np.round(columns[key], decimal)

    if asterisks_dict:
        columns["Signif"] = map(
            add_asterisks, columns[asterisks_dict["pvalue"]]
        )

    df = pd.DataFrame.from_dict(columns)
    df.index += index_start
    return df


def add_asterisks(pvalue):
    asterisks = ""
    if pvalue < 0.01:
        asterisks = "***"
    elif pvalue < 0.05:
        asterisks = "**"
    elif pvalue < 0.1:
        asterisks = "*"
    return asterisks


def plot(time, CAR, *, AR=None, CI=False, var=None, df=None, confidence=0.95):

    fig, ax = plt.subplots()
    ax.plot(time, CAR)
    ax.axvline(
        x=0, color="black", linewidth=0.5,
    )
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if CI:
        delta = np.sqrt(var) * t.ppf(confidence, df)
        upper = CAR + delta
        lower = CAR - delta
        ax.fill_between(time, lower, upper, color="black", alpha=0.1)

    if AR is not None:
        ax.vlines(time, ymin=0, ymax=AR)

    if ax.get_ylim()[0] * ax.get_ylim()[1] < 0:
        # if the y axis contains 0
        ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")

    return fig


def plot_dist(X, scale=1, seed=3836):
    sns.distplot(X)
    np.random.seed(seed)
    xn = np.random.normal(scale=0.25, size=1000) 
    fig = sns.distplot(xn, kde=True, hist=False)
    return fig


def get_date_idx(X, date: np.datetime64, n: int = 5):
    idx = None
    for i in range(n):
        index = np.where(X == date)[0]
        if len(index) > 0:
            idx =  index[0]
        else:
            date += np.timedelta64(1, "D")
    return idx

def get_logreturns():
    try:
        storage_options = {}
        storage_options['AWS_REGION'] = 'us-east-1'
        try:
            # this is for iam role authentication on the backtest box
            session = boto3.session.Session()
            credentials = session.get_credentials()
            storage_options["AWS_ACCESS_KEY_ID"] = credentials.access_key
            storage_options["AWS_SECRET_ACCESS_KEY"] = credentials.secret_key
            storage_options["AWS_SESSION_TOKEN"] = credentials.token
        except:
            # this is for local user auth
            sts = boto3.client('sts')
            token = sts.get_session_token()
            credentials = token.get('Credentials')
            storage_options["AWS_ACCESS_KEY_ID"] = credentials.get('AccessKeyId')
            storage_options["AWS_SECRET_ACCESS_KEY"] = credentials.get('SecretAccessKey')
            storage_options["AWS_SESSION_TOKEN"] = credentials.get('SessionToken')

        data = None
        
        f = {}
        db = pgdb2.database(mode="sessrw")
        engine, conn, _ = db.getEngineConnCursor()

        # read market_tradedays
        ddf = pd.read_sql(
            """
            select date,
                idxdate
            from market_tradedays
            where date >= '2003-07-01'
                and date < current_date
            """,
            con=engine)
        f['market_tradedays'] = ddf

        # read market returns
        mdf = pd.read_sql(
            """
            set statement_timeout=0;
            select md.date,
                i.group_id::text as industry_id,
                case when yiv.close <> 0 then 
                        round(((iv.close / yiv.close) - 1)::numeric, 6) 
                    else
                        0 
                    end as avgreturn
            from indexes i
            inner join indexvalues iv 
                on iv.id = i.id
            inner join indexvalues yiv 
                on yiv.id = iv.id 
                    and yiv.date = getmarketdate(iv.date - interval '1 day')
            inner join market_tradedays md 
                on md.date = iv.date
            order by md.date, 
                industry_id
            """,
            con=engine)
        conn.close()

        # convert daily returns to log returns
        mdf['avg_logreturn'] = np.log(1 + mdf['avgreturn'])
        # datetime64 the date
        mdf['date'] = mdf['date'].astype(np.datetime64)
        # pivot returns so each id is a column
        mktret_df = mdf.pivot(index=["date"], columns=["industry_id"], values=["avgreturn"]).reset_index()
        cols = mktret_df.columns.get_level_values(1).tolist()
        cols[0] = 'date'
        mktret_df.columns = cols
        f["mkt_logreturns"] = mktret_df

        # read eodquotes delta table
        quotes_uri = "s3://eventstudy/quotes.delta"
        dt = DeltaTable(quotes_uri, storage_options=storage_options)
        qdf = dt.to_pandas(columns=['cid', 'trade_date', 'dailyreturn'])

        #convert daily returns to log returns
        qdf['logdailyreturn'] = np.log(1 + qdf['dailyreturn'])
        # datetime64 the date
        qdf['trade_date'] = qdf['trade_date'].astype(np.datetime64)
        
        # pivot so each cid is a column
        quotes = qdf.pivot(index="trade_date", columns="cid", values="logdailyreturn")
        quotes = quotes.fillna(0)
        quotes.index.rename("date", inplace=True)
        f["logreturns"] = quotes

        # merge market returns with quotes delta table
        df = pd.merge(f["market_tradedays"], f["mkt_logreturns"], how="inner", on=["date"])
        data = pd.merge(df, f["logreturns"], how="inner", on=["date"])
        data.columns = [str(c) for c in data.columns]
    except Exception:
        msg = sys.exc_info()
        details = traceback.format_exc()
        msg = f"{msg}: {details}"
        print(msg)

    return data

def update_famafrench():
    db = pgdb2.database(mode='SESSRW')
    engine, conn, _ = db.getEngineConnCursor()

    models = {
        'F-F_Research_Data_Factors_daily': 'ff3',
        'F-F_Research_Data_5_Factors_2x3_daily': 'ff5',
        'F-F_Momentum_Factor_daily': 'momentum'}

    odf = None
    for model in models.keys():
        data = pdr.DataReader(model, 'famafrench', start='2003-07-01')
        df = data[0]
        if models[model] == 'ff3':
            cols = {
                'Mkt-RF': 'mkt_rf',
                'SMB': 'ff3_smb',
                'HML': 'ff3_hml',
                'RF': 'ff3_rf'}
        elif models[model] == 'ff5':
            cols = {
                'Mkt-RF': 'mkt_rf',
                'SMB': 'ff5_smb',
                'HML': 'ff5_hml',
                'RMW': 'ff5_rmw',
                'CMA': 'ff5_cma',
                'RF': 'ff5_rf'}
        else:
            cols = {'Mom   ': 'mom'}
        df.rename(columns=cols, inplace=True)
        if models[model] == 'ff5':
            df.drop(columns=['mkt_rf'], inplace=True)
        if odf is None:
            odf = df
        else: 
            odf = pd.concat([odf, df], axis=1)
    odf.index.rename("date", inplace=True)
    odf.to_sql('famafrench', schema='quotes', if_exists='replace', con=engine)
    conn.close()

def insert_quotes_history(date: np.datetime64=np.datetime64(datetime.date.today()), s3: bool=True):
    sts = boto3.client('sts')
    token = sts.get_session_token()
    credentials = token.get('Credentials')
    credentials

    storage_options = None
    if s3:
        storage_options = {}
        storage_options["AWS_ACCESS_KEY_ID"] = credentials.get('AccessKeyId')
        storage_options["AWS_SECRET_ACCESS_KEY"] = credentials.get('SecretAccessKey')
        storage_options["AWS_SESSION_TOKEN"] = credentials.get('SessionToken')
        storage_options["AWS_S3_ALLOW_UNSAFE_RENAME"] = 'true'
        storage_options['AWS_REGION'] = 'us-east-1'

    warnings.simplefilter(action='ignore', category=Warning)

    db = pgdb2.database()
    engine, conn, cursor = db.getEngineConnCursor()

    if date is None or date < np.datetime64("2003-07-01") :
        sql = """
            select date
            from market_tradedays
            where date >= '2003-07-01'
                and date < current_date
            """
        df = pd.read_sql(sql, con=conn)
        dates = df['date'].to_list()
    elif date <= np.datetime64(datetime.date.today()):
        sql = """
            select getmarketdate(%s::date - interval '1 day') as date
            """
        cursor.execute(sql, vars=[date.astype(datetime.date)])
        date = cursor.fetchone()['date']
        dates = [date]
    else:
        raise FutureDateError(date)

    quotes_sql = """
        select cid,
            shareclassid,
            cusip,
            ticker,
            trade_date,
            open,
            high,
            low,
            close,
            volume,
            adjopen,
            adjhigh,
            adjlow,
            adjclose,
            adjvolume,
            sharesout,
            avg30dayvolume,
            edited,
            ignored,
            cumprice_factor,
            cumshares_factor,
            divamount,
            dailygain,
            dailyreturn
        from quotes
        where trade_date = %s
        order by cid
        """
    
    quotes_dtypes = {
        "security_ticker": np.str_,
        "trade_date": np.datetime64,
        "open": np.float64,
        "high": np.float64,
        "low": np.float64,
        "close": np.float64,
        "adjclose": np.float64,
        "volume": np.float64,
    }

    i = 0    
    for dt in dates:
        qdf = pd.read_sql_query(quotes_sql, con=engine, params=[dt], dtype=quotes_dtypes)
        qdf['trade_date'] = pd.to_datetime(qdf['trade_date']).dt.date
        i += 1
        if i > 0 or len(dates) == 1:
            mode = 'append'
        else:
            mode = 'overwrite'
        s3_path = ""
        if s3:
            s3_path = "s3://eventstudy/"
        write_deltalake(f'{s3_path}quotes.delta', qdf, mode=mode, storage_options=storage_options)
        print(dt)