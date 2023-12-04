import datetime
import pyarrow.parquet as pq
import pyarrow.fs as pfs
import numpy as np
import pandas as pd
from scipy.stats import t
import pgdb2
import s3fs
import warnings


from .utils import (
    to_table, 
    plot, 
    get_date_idx,
    get_logreturns,
    update_famafrench)


from .exception import (
    ParameterMissingError,
    DateMissingError,
    DataMissingError,
    ColumnMissingError,
    ReturnsCacheEmptyError,
    EventFormatError,
    EventKeyError)

from .models import (
    ordinary_returns_model,
    market_adjusted_model,
    mean_adjusted_model,
    market_model,
    fama_french_3,
    fama_french_5,
    carhart)

warnings.simplefilter(action='ignore', category=Warning)


class Single:
    """
    Event Study package's core object. Implement the classical event study methodology [1]_ for a single event.
    This implementation heavily relies on the work of MacKinlay [2]_.

    References
    ----------

    .. [1] Fama, E. F., L. Fisher, M. C. Jensen, and R. Roll (1969). 
        “The Adjustment of Stock Prices to New Information”.
        In: International Economic Review 10.1, pp. 1–21.
    .. [2] Mackinlay, A. (1997). “Event Studies in Economics and Finance”.
        In: Journal of Economic Literature 35.1, p. 13.
    """

    _parameters = {
        "max_iteration": 5,
    }

    def __init__(
        self,
        model_func,
        model_data: dict,
        security_ticker: int = None,
        market_ticker: int = None,
        event_date: np.datetime64 = None,
        event_window: tuple = (-5, +5),
        estimation_size: int = 252,
        buffer_size: int = 21,
        weight: int = 1,
        description: str = None
    ):
        """
        Parameters
        ----------
        model_func
            Function computing the modelisation of returns.
        model_data : dict
            Dictionary containing all parameters needed by `model_func`.
        event_date : np.datetime64
            Date of the event in numpy.datetime64 format.
        event_window : tuple, optional
            Event window specification (T2,T3), by default (-5, +5).
            A tuple of two integers, representing the start and the end of the event window. 
            Classically, the event-window starts before the event and ends after the event.
            For example, `event_window = (-2,+20)` means that the event-period starts
            2 periods before the event and ends 20 periods after.
        estimation_size : int, optional
            Size of the estimation for the modelisation of returns [T0,T1], by default 252
        buffer_size : int, optional
            Size of the buffer window [T1,T2], by default 21
        weight : int, optional
            Weight to be applied to the returns in the MultipleEvents Object

        Example
        -------
        Run an event study based on :
        .. the `market_model` function defined in the `models` submodule,
        .. given values for security and market returns,
        .. and default parameters

        >>> from eventstudy import Single, models
        >>> event = Single(
        ...     models.market_model, 
        ...     {'security_returns':[0.032,-0.043,...], 'market_returns':[0.012,-0.04,...]}
        ... )
        """
        self.security_ticker = security_ticker
        self.market_ticker = market_ticker,
        self.event_date = event_date
        self.event_window = event_window
        self.event_window_size = -event_window[0] + event_window[1] + 1
        self.estimation_size = estimation_size
        self.buffer_size = buffer_size
        self.weight = weight
        self.description = description

        model = model_func(
            **model_data,
            estimation_size=self.estimation_size,
            event_window_size=self.event_window_size)

        self.AR, self.df, self.var_AR, self.model = model
        self.__compute()

    def __compute(self):
        self.CAR = np.cumsum(self.AR)
        self.var_CAR = [(i * var) for i, var in enumerate(self.var_AR, 1)]
        self.tstat = self.CAR / np.sqrt(self.var_CAR)
        # see https://www.statology.org/t-distribution-python/
        self.pvalue = (1.0 - t.cdf(abs(self.tstat), self.df)) * 2

    def results(self, asterisks: bool = True, decimals=4):
        """
        Return event study's results in a table format.
        
        Parameters
        ----------
        asterisks : bool, optional
            Add asterisks to CAR value based on significance of p-value, by default True
        decimals : int or list, optional
            Round the value with the number of decimal specified, by default 3.
            `decimals` can either be an integer, in this case all value will be 
            round at the same decimals, or a list of 6 decimals, in this case each 
            columns will be round based on its respective number of decimal.
        
        Note
        ----

        When `asterisks` is set as True, CAR's are converted to string type.
        To make further computation on CARs possible set `asterisks` to False.

        Returns
        -------
        pandas.DataFrame
            AR and AR's variance, CAR and CAR's variance, T-stat and P-value, 
            for each T in the event window.

        Note
        ----
        
        The function return a fully working pandas DataFrame.
        All pandas method can be used on it, especially exporting method (to_csv, to_excel,...)

        Example
        -------

        Get results of a market model event study, with specific number of decimal for each column:

        >>> event = EventStudy.market_model(
        ...     security_ticker = 'AAPL',
        ...     market_ticker = 'SPY',
        ...     event_date = np.datetime64('2007-01-09'),
        ...     event_window = (-5,+5)
        ... )
        >>> event.results(decimals = [3,5,3,5,2,2])

        Note
        ----
        
        Significance level: \*\*\* at 99%, \*\* at 95%, \* at 90%
        """

        columns = {
            "AR": self.AR,
            "Std. E. AR": np.sqrt(self.var_AR),
            "CAR": self.CAR,
            "Std. E. CAR": np.sqrt(self.var_CAR),
            "T-stat": self.tstat,
            "P-value": self.pvalue,
        }

        asterisks_dict = {"pvalue": "P-value"} if asterisks else None
        
        return to_table(
            columns,
            asterisks_dict=asterisks_dict,
            decimals=decimals,
            index_start=self.event_window[0],
        )

    def plot(self, *, AR=False, CI=True, confidence=0.95):
        """
        Plot the event study result.
        
        Parameters
        ----------
        AR : bool, optional
            Add to the figure a bar plot of AR, by default False
        CI : bool, optional
            Display the confidence interval, by default True
        confidence : float, optional
            Set the confidence level, by default 0.95
        
        Returns
        -------
        matplotlib.figure
            Plot of CAR and AR (if specified).

        Note
        ----
        The function return a fully working matplotlib function.
        You can extend the figure and apply new set-up with matplolib's method (e.g. savefig).
        
        Example
        -------

        Plot CAR (in blue) and AR (in black), with a confidence interval of 95% (in grey).

        >>> event = EventStudy.market_model(
        ...     security_ticker = 'AAPL',
        ...     market_ticker = 'SPY',
        ...     event_date = np.datetime64('2007-01-09'),
        ...     event_window = (-5,+20)
        ... )
        >>> event.plot(AR = True, confidence = .95)

        .. image:: /_static/single_event_plot.png
        """

        return plot(
            time=range(self.event_window[0], self.event_window[1] + 1),
            CAR=self.CAR,
            AR=self.AR if AR else None,
            CI=CI,
            var=self.var_CAR,
            df=self.df,
            confidence=confidence,
        )

    @classmethod
    def _save_parameter(cls, param_name: str, data):
        cls._parameters[param_name] = data

    @classmethod
    def _get_parameters(
        cls,
        param_name: str,
        columns: tuple,
        event_date: np.datetime64,
        event_window: tuple = (-5, +5),
        estimation_size: int = 252,
        buffer_size: int = 21,
        weight: int = 1
    ) -> tuple:

        # Find index of returns
        try:
            event_i = get_date_idx(
                cls._parameters[param_name]["date"],
                event_date,
                cls._parameters["max_iteration"],
            )
        except KeyError:
            raise ParameterMissingError(param_name)

        if event_i is None:
            raise DateMissingError(event_date, param_name)

        start = event_i - (-event_window[0] + buffer_size + estimation_size)
        end = event_i + event_window[1] + 1
        size = -event_window[0] + buffer_size + estimation_size + event_window[1] + 1

        results = list()
        for column in columns:
            try:
                result = cls._parameters[param_name][column][start:end]
            except KeyError:
                raise ColumnMissingError(param_name, column)

            # test if all data has been retrieved
            if len(result) != size:
                msg = ", ".join([
                    f"event_date: {event_date}",
                    f"event_idx: {event_i}",
                    f"result: {len(result)}", 
                    f"size: {size}",
                    f"start: {start}",
                    f"end: {end}",
                    f"weight: {weight}"])
                raise DataMissingError(param_name, column, len(result), start + end, msg)

            results.append(result)

        return tuple(results)

    @classmethod
    def import_returns(
        cls,
        *,
        path=None,
        cached=True
    ):
        """
        Import returns from a delta file for the `SingleEvent` Class parameters.
        Delta file is in the form of date, [market_ticker, sector_idx, ...], [security_tickers]
        Once imported, the returns are shared among all `SingleEvent` instances.

        
        Parameters
        ----------
        path : str
            Path to the cross sectional returns' parquet file
            Not providing a path will build/return the standard file based upond
            the cached parameter
        cached : boolean
            Load the standard cache
        """
        if path is None:
            dt = datetime.date.today().strftime("%Y%m%d")
            lr_base_uri = f"eventstudy/logreturns-{dt}.parquet"
            fs = s3fs.S3FileSystem(anon=False)
            lr_uri = f"s3://{lr_base_uri}"
            if cached:
                try:
                    files = fs.glob(lr_base_uri)
                    if len(files) > 0:
                        print(f"Cached Returns Found: {lr_base_uri}")
                        raw_fs, normalized_path = pfs.FileSystem.from_uri(lr_uri)
                        filesystem = pfs.SubTreeFileSystem(normalized_path, raw_fs)
                        # load the file if it exists
                        data = pq.read_table(lr_uri).to_pandas()
                        print(f"Cached Returns: {data.shape}")
                    else:
                        print("Cache Empty")
                        raise ReturnsCacheEmptyError
                except ReturnsCacheEmptyError:
                    print("Getting Log Returns...")
                    data = get_logreturns()
                    print(f"Got Log Returns: {data.shape}")
                    print("Caching Log Returns...")
                    data.to_parquet(lr_uri)
                    print(f"Cached Log Returns: {lr_uri}")
            else:
                print("Getting Log Returns...")
                data = get_logreturns()
                print(f"Got Log Returns: {data.shape}")
        else:
            data = data = pq.read_table(path).to_pandas()
        if 'M1' in data.columns:
            data['M99'] = data["M1"] 
        data.fillna(0, inplace=True)
        data.replace([np.inf, -np.inf], 0, inplace=True)
        cls._save_parameter("returns", data)
    

    @classmethod
    def import_FamaFrench(cls, update=False):
        if update:
            update_famafrench()
        db = pgdb2.database(mode='sessro')
        engine, conn, _ = db.getEngineConnCursor()
        sql = """
            select date, 
                mkt_rf,
                ff3_smb,
                ff3_hml,
                ff3_rf,
                ff5_smb,
                ff5_hml,
                ff5_rmw,
                ff5_cma,
                ff5_rf,
                mom
            from quotes.famafrench
            order by date
            """
        data = pd.read_sql(sql, con=engine)
        conn.close()
        for key in data.columns:
            if key != "date":
                data[key] = data[key] / 100

        cls._save_parameter("FamaFrench", data)

    @classmethod
    def filter_event_returns(
        cls,
        df,
        event_window: tuple = (-5, +5),
        est_size: int = 252,
        buffer_size: int = 30,
        famafrench: bool = False
    ):
        """
        convenience method to filter events using the returns 
        table so it doesn't have to be manually figured out
        df can be a pandas dataframe or a list of dict
        the method returns a tuple of pandas dataframes
        a filtered dataframe of events, and an excluded dataframe
        of events
        """
        colkeys = ('security_ticker', 'event_date')
        if type(df) == list:
            if len(df) == 0:
                return df
            else:
                for d in df:
                    if type(d) != dict:
                        raise EventFormatError
                    else:
                        if not all(k in d for k in colkeys):
                            for k in colkeys:
                                if k not in d:
                                    raise EventKeyError(k)
            df = pd.DataFrame(df)
        else:
            if not all(k in df.columns for k in colkeys):
                raise EventKeyError(k)
            
        if famafrench:
            min_ff = cls._parameters['FamaFrench']['date'].min()
            max_ff = cls._parameters['FamaFrench']['date'].max()
            df = df[
                (df['event_date'] >= min_ff) 
                & (df['event_date'] <= max_ff)]
            
        start_retidx = np.max([est_size + buffer_size - event_window[0], 0])
        end_retidx = event_window[1]
        min_retdate = cls._parameters['returns']['date'].iloc[start_retidx]
        max_retdate = cls._parameters['returns']['date'].iloc[-end_retidx]
        dates = cls._parameters['returns']['date']
        colnames = cls._parameters['returns'].columns

        market_filter = None
        if 'market_ticker' in df.columns:
            market_filter = df['market_ticker'].astype(str).isin(colnames)
        
        df_filter = (
            (df['event_date'] >= min_retdate) 
            & (df['event_date'] <= max_retdate) 
            & (df['event_date'].isin(dates))
            & (df['security_ticker'].astype(str).isin(colnames)))
        
        if market_filter is not None:
            df_filter = (df_filter) & (market_filter)

        df = df[df_filter]
        exclude_df = df[~df_filter]
        return (df, exclude_df)

    @classmethod
    def market_model(
        cls,
        security_ticker: str,
        market_ticker: str,
        event_date: np.datetime64,
        event_window: tuple = (-5, +5),
        estimation_size: int = 252,
        buffer_size: int = 21,
        weight: int = 1,
        **kwargs
    ):
        """
        Modelise returns with the market model.
        
        Parameters
        ----------
        security_ticker : str
            Ticker of the security (e.g. company stock) as given in the returns imported.
        market_ticker : str
            Ticker of the market (e.g. market index) as given in the returns imported.
        event_date : np.datetime64
            Date of the event in numpy.datetime64 format.
        event_window : tuple, optional
            Event window specification (T2,T3), by default (-10, +10).
            A tuple of two integers, representing the start and the end of the event window. 
            Classically, the event-window starts before the event and ends after the event.
            For example, `event_window = (-2,+20)` means that the event-period starts
            2 periods before the event and ends 20 periods after.
        estimation_size : int, optional
            Size of the estimation for the modelisation of returns [T0,T1], by default 300
        buffer_size : int, optional
            Size of the buffer window [T1,T2], by default 30
        weight : int, optional
            Weight to be applied to the returns in the MultipleEvents Object
        **kwargs
            Additional keywords have no effect but might be accepted to avoid freezing 
            if there are not needed parameters specified.
        
        See also
        -------
        
        fama_french_3, mean_adjusted_model

        Example
        -------

        Run an event study for the Apple company for the announcement of the first iphone,
        based on the market model with the S&P500 index as a market proxy.

        >>> event = EventStudy.market_model(
        ...     security_ticker = 'AAPL',
        ...     market_ticker = 'SPY',
        ...     event_date = np.datetime64('2007-01-09'),
        ...     event_window = (-5,+20)
        ... )
        """
        security_returns, market_returns = cls._get_parameters(
            "returns",
            (security_ticker, market_ticker),
            event_date,
            event_window,
            estimation_size,
            buffer_size,
            weight
        )
        description = f"Market model estimation, Security: {security_ticker}, Market: {market_ticker}"

        return cls(
            market_model,
            {"security_returns": security_returns, "market_returns": market_returns},
            event_window=event_window,
            estimation_size=estimation_size,
            buffer_size=buffer_size,
            description= description,
            security_ticker=security_ticker,
            event_date=event_date,
            weight=weight
        )
        
    @classmethod
    def market_adjusted_model(
        cls,
        security_ticker : str,
        market_ticker: str,
        event_date: np.datetime64,
        event_window: tuple = (-5, +5),
        est_size: int = 252,
        buffer_size: int = 30,
        weight: int = 1
    ):
        """
        Model the returns with the market model.
        
        Parameters
        ----------
        security_ticker  : str
            security_ticker  of the returns imported.
        market_ticker : str
            market ticker of the returns imported.
        event_date : np.datetime64
            Date of the event in numpy.datetime64 format.
        event_window : tuple, optional
            Event window specification (T2,T3), by default (-5, +5).
            A tuple of two integers, representing the start and the end of the event window. 
            Classically, the event-window starts before the event and ends after the event.
            For example, `event_window = (-2,+20)` means that the event-period starts
            2 periods before the event and ends 20 periods after.
        est_size : int, optional
            Size of the estimation for the modelisation of returns [T0,T1], by default 252
        buffer_size : int, optional
            Size of the buffer window [T1,T2], by default 21
        weight : int, optional
            Weight to be applied to the returns in the MultipleEvents Object
        **kwargs
            Additional keywords have no effect but might be accepted to avoid freezing 
            if there are not needed parameters specified.
        
        Example
        -------
        Run an event study for the Apple company for the announcement of the first iphone,
        based on the market model with the S&P500 index as a market proxy.
        >>> event = SingleEvent.MarketModel(
        ...     security_ticker='AAPL',
        ...     market_ticker=2113,
        ...     event_date=np.datetime64('2007-01-09'),
        ...     event_window=(-5,+20)
        ... )
        """
        daily_ret, daily_mkt = cls._get_parameters(
            "returns",
            (security_ticker, market_ticker,),
            event_date,
            event_window,
            est_size,
            buffer_size,
            weight
        )
        description = f"Market model estimation, security_ticker: {security_ticker}, Market: {market_ticker}"

        return cls(
            market_adjusted_model,
            {"daily_ret": daily_ret, "daily_mkt": daily_mkt},
            event_window=event_window,
            est_size=est_size,
            buffer_size=buffer_size,
            description= description,
            security_ticker=security_ticker,
            event_date=event_date,
            weight=weight
        )

    @classmethod
    def mean_adjusted_model(
        cls,
        security_ticker,
        event_date: np.datetime64,
        event_window: tuple = (-5, +5),
        estimation_size: int = 252,
        buffer_size: int = 21,
        weight: int = 1,
        **kwargs
    ):
        """
        Model the returns with the mean adjusted model.
        
        Parameters
        ----------
        security_ticker : str
            Ticker of the security (e.g. company stock) as given in the returns imported.
        event_date : np.datetime64
            Date of the event in numpy.datetime64 format.
        event_window : tuple, optional
            Event window specification (T2,T3), by default (-10, +10).
            A tuple of two integers, representing the start and the end of the event window. 
            Classically, the event-window starts before the event and ends after the event.
            For example, `event_window = (-2,+20)` means that the event-period starts
            2 periods before the event and ends 20 periods after.
        estimation_size : int, optional
            Size of the estimation for the modelisation of returns [T0,T1], by default 300
        buffer_size : int, optional
            Size of the buffer window [T1,T2], by default 30
        weight : int, optional
            Weight to be applied to the returns in the MultipleEvents Object
        **kwargs
            Additional keywords have no effect but might be accepted to avoid freezing 
            if there are not needed parameters specified.
            For example, if market_ticker is specified.
        
        See also
        -------
        market_model, Single.FamaFrench_3factor

        Example
        -------

        Run an event study for the Apple company for the announcement of the first iphone,
        based on the constant mean model.

        >>> event = EventStudy.mean_adjusted_model(
        ...     security_ticker = 'AAPL',
        ...     event_date = np.datetime64('2007-01-09'),
        ...     event_window = (-5,+20)
        ... )
        """
        # the comma after 'security_returns' unpack the one-value tuple returned by the function _get_parameters
        (security_returns,) = cls._get_parameters(
            "returns",
            (security_ticker,),
            event_date,
            event_window,
            estimation_size,
            buffer_size,
            weight
        )
        
        description = f"Mean Adjusted Model estimation, Security: {security_ticker}"
        
        return cls(
            mean_adjusted_model,
            {"security_returns": security_returns},
            event_window=event_window,
            estimation_size=estimation_size,
            buffer_size=buffer_size,
            description=description,
            security_ticker=security_ticker,
            event_date=event_date,
            weight=weight
        )
        
    @classmethod
    def ordinary_returns_model(
        cls,
        security_ticker,
        event_date: np.datetime64,
        event_window: tuple = (-5, +5),
        est_size: int = 252,
        buffer_size: int = 21,
        weight: int = 1,
        **kwargs
    ):
        """
        Model the returns with the ordinary returns model.
        
        Parameters
        ----------
        security_ticker  : str
            security_ticker  of the returns imported.
        event_date : np.datetime64
            Date of the event in numpy.datetime64 format.
        event_window : tuple, optional
            Event window specification (T2,T3), by default (-5, +5).
            A tuple of two integers, representing the start and the end of the event window. 
            Classically, the event-window starts before the event and ends after the event.
            For example, `event_window = (-2,+20)` means that the event-period starts
            2 periods before the event and ends 20 periods after.
        est_size : int, optional
            Size of the estimation for the modelisation of returns [T0,T1], by default 252
        buffer_size : int, optional
            Size of the buffer window [T1,T2], by default 21
        weight : int, optional
            Weight to be applied to the returns in the MultipleEvents Object
        **kwargs
            Additional keywords have no effect but might be accepted to avoid freezing 
            if there are not needed parameters specified.
        
        Example
        -------
        Run an event study for the Apple company for the announcement of the first iphone,
        based on the mean adjusted model.
        >>> event = SingleEvent.RawReturnsModel(
        ...     security_ticker  = 'AAPL'',
        ...     event_date = np.datetime64('2007-01-09'),
        ...     event_window = (-5,+20)
        ... )
        """
        # the comma after 'daily_ret' unpack the one-value tuple returned by the function _get_parameters
        (daily_ret,) = cls._get_parameters(
            "returns",
            (security_ticker,),
            event_date,
            event_window,
            est_size,
            buffer_size,
            weight
        )
        description = f"Raw Returns, security_ticker : {security_ticker}"
        
        return cls(
            ordinary_returns_model,
            {"daily_ret": daily_ret},
            event_window=event_window,
            est_size=est_size,
            buffer_size=buffer_size,
            description=description,
            security_ticker=security_ticker,
            event_date=event_date,
            weight=weight
        )

    @classmethod
    def fama_french_3(
        cls,
        security_ticker,
        event_date: np.datetime64,
        event_window: tuple = (-5, +5),
        estimation_size: int = 252,
        buffer_size: int = 21,
        weight: int = 1,
        **kwargs
    ):
        """
        Modelise returns with the Fama-French 3-factor model.
        The model used is the one developped in Fama and French (1992) [1]_.
        
        Parameters
        ----------
        security_ticker : str
            Ticker of the security (e.g. company stock) as given in the returns imported.
        event_date : np.datetime64
            Date of the event in numpy.datetime64 format.
        event_window : tuple, optional
            Event window specification (T2,T3), by default (-5, +5).
            A tuple of two integers, representing the start and the end of the event window. 
            Classically, the event-window starts before the event and ends after the event.
            For example, `event_window = (-2,+20)` means that the event-period starts
            2 periods before the event and ends 20 periods after.
        estimation_size : int, optional
            Size of the estimation for the modelisation of returns [T0,T1], by default 252
        buffer_size : int, optional
            Size of the buffer window [T1,T2], by default 21
        keep_model : bool, optional
            If true the model used to compute the event study will be stored in memory.
            It will be accessible through the class attributes eventstudy.Single.model, by default False
        **kwargs
            Additional keywords have no effect but might be accepted to avoid freezing 
            if there are not needed parameters specified.
            For example, if market_ticker is specified.
        
        See also
        -------
        market_model, mean_adjusted_model

        Example
        -------

        Run an event study for the Apple company for the announcement of the first iphone,
        based on the Fama-French 3-factor model.

        >>> event = EventStudy.fama_french_3(
        ...     security_ticker = 'AAPL',
        ...     event_date = np.datetime64('2007-01-09'),
        ...     event_window = (-5,+20)
        ... )

        References
        ----------
        .. [1] Fama, E. F. and K. R. French (1992). 
            “The Cross-Section of Expected Stock Returns”.
            In: The Journal of Finance 47.2, pp. 427–465.
        """

        (security_returns,) = cls._get_parameters(
            "returns",
            (security_ticker,),
            event_date,
            event_window,
            estimation_size,
            buffer_size,
            weight
        )
        Mkt_RF, SMB, HML, RF = cls._get_parameters(
            "FamaFrench",
            ("mkt_rf", "ff3_smb", "ff3_hml", "ff3_rf"),
            event_date,
            event_window,
            estimation_size,
            buffer_size,
            weight
        )
        
        description = f"Fama-French 3-factor model estimation, Security: {security_ticker}"
        
        return cls(
            fama_french_3,
            {
                "security_returns": security_returns,
                "Mkt_RF": Mkt_RF,
                "SMB": SMB,
                "HML": HML,
                "RF": RF,
            },
            event_window=event_window,
            estimation_size=estimation_size,
            buffer_size=buffer_size,
            description=description,
            security_ticker=security_ticker,
            event_date=event_date,
            weight=weight
        )

    @classmethod
    def fama_french_5(
        cls,
        security_ticker,
        event_date: np.datetime64,
        event_window: tuple = (-5, +5),
        estimation_size: int = 252,
        buffer_size: int = 21,
        weight: int = 1,
        **kwargs
    ):
        """
        Modelise returns with the Fama-French 5-factor model.
        The model used is the one developped in Fama and French (1992) [1]_.
        
        Parameters
        ----------
        security_ticker : str
            Ticker of the security (e.g. company stock) as given in the returns imported.
        event_date : np.datetime64
            Date of the event in numpy.datetime64 format.
        event_window : tuple, optional
            Event window specification (T2,T3), by default (-10, +10).
            A tuple of two integers, representing the start and the end of the event window. 
            Classically, the event-window starts before the event and ends after the event.
            For example, `event_window = (-2,+20)` means that the event-period starts
            2 periods before the event and ends 20 periods after.
        estimation_size : int, optional
            Size of the estimation for the modelisation of returns [T0,T1], by default 300
        buffer_size : int, optional
            Size of the buffer window [T1,T2], by default 30
        keep_model : bool, optional
            If true the model used to compute the event study will be stored in memory.
            It will be accessible through the class attributes eventstudy.Single.model, by default False
        **kwargs
            Additional keywords have no effect but might be accepted to avoid freezing 
            if there are not needed parameters specified.
            For example, if market_ticker is specified.
        
        See also
        -------
        market_model, mean_adjusted_model

        Example
        -------

        Run an event study for the Apple company for the announcement of the first iphone,
        based on the Fama-French 5-factor model.

        >>> event = EventStudy.fama_french_5(
        ...     security_ticker = 'AAPL',
        ...     event_date = np.datetime64('2007-01-09'),
        ...     event_window = (-5,+20)
        ... )

        References
        ----------
        .. [1] Fama, E. F. and K. R. French (1992). 
            “The Cross-Section of Expected Stock Returns”.
            In: The Journal of Finance 47.2, pp. 427–465.
        """

        (security_returns,) = cls._get_parameters(
            "returns",
            (security_ticker,),
            event_date,
            event_window,
            estimation_size,
            buffer_size,
        )
        Mkt_RF, SMB, HML, RMW, CMA, RF = cls._get_parameters(
            "FamaFrench",
            ("mkt_rf", "ff5_smb", "ff5_hml", "ff5_rmw", "ff5_cma", "ff5_rf"),
            event_date,
            event_window,
            estimation_size,
            buffer_size,
            weight
        )
        
        description = f"Fama-French 5-factor model estimation, Security: {security_ticker}"
        
        return cls(
            fama_french_5,
            {
                "security_returns": security_returns,
                "Mkt_RF": Mkt_RF,
                "SMB": SMB,
                "HML": HML,
                "RMW": RMW,
                "CMA": CMA,
                "RF": RF,
            },
            event_window=event_window,
            estimation_size=estimation_size,
            buffer_size=buffer_size,
            description=description,
            security_ticker=security_ticker,
            event_date=event_date,
            weight=weight
        )
        
    @classmethod
    def carhart(
        cls,
        security_ticker,
        event_date: np.datetime64,
        event_window: tuple = (-5, +5),
        est_size: int = 252,
        buffer_size: int = 21,
        weight: int = 1,
        **kwargs
    ):
        """
        Model the returns with the Carhart (Fama-French 3-factor + Momentum) model.
        
        Parameters
        ----------
        security_ticker : str
            security_ticker of the returns imported.
        event_date : np.datetime64
            Date of the event in numpy.datetime64 format.
        event_window : tuple, optional
            Event window specification (T2,T3), by default (-5, +5).
            A tuple of two integers, representing the start and the end of the event window. 
            Classically, the event-window starts before the event and ends after the event.
            For example, `event_window = (-2,+20)` means that the event-period starts
            2 periods before the event and ends 20 periods after.
        est_size : int, optional
            Size of the estimation for the modelisation of returns [T0,T1], by default 252
        buffer_size : int, optional
            Size of the buffer window [T1,T2], by default 21
        weight : int, optional
            Weight to be applied to the returns in the MultipleEvents Object
        **kwargs
            Additional keywords have no effect but might be accepted to avoid freezing 
            if there are not needed parameters specified.

        Example
        -------
        Run an event study for the Apple company for the announcement of the first iphone,
        based on the Carhart model.
        >>> event = SingleEvent.Carhart(
        ...     security_ticker = 'AAPL',
        ...     event_date = np.datetime64('2007-01-09'),
        ...     event_window = (-5,+20)
        ... )
        """

        (daily_ret,) = cls._get_parameters(
            "returns",
            (security_ticker,),
            event_date,
            event_window,
            est_size,
            buffer_size,
            weight
        )
        Mkt_RF, SMB, HML, RF, MOM = cls._get_parameters(
            "FamaFrench",
            ("mkt_rf", "ff3_smb", "ff3_hml",  "ff3_rf", "mom"),
            event_date,
            event_window,
            est_size,
            buffer_size,
            weight
        )

        description = f"Carhart model estimation, security_ticker: {security_ticker}"
        
        return cls(
            carhart,
            {
                "daily_ret": daily_ret,
                "Mkt_RF": Mkt_RF,
                "SMB": SMB,
                "HML": HML,
                "RF": RF,
                "MOM": MOM,
            },
            event_window=event_window,
            est_size=est_size,
            buffer_size=buffer_size,
            description=description,
            security_ticker=security_ticker,
            event_date=event_date,
            weight=weight
        )