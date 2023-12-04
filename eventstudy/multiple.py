from .utils import (
    to_table, 
    plot)

from .exception import (
    DateMissingError,
    DataMissingError,
    ColumnMissingError,
)
import logging
import pandas as pd
import numpy as np
from scipy.stats import t, kurtosis, skew
from typing import Callable
from pyod.utils.utility import get_label_n


class Multiple:
    """
    Implement computations on an aggregate of event studies using average 
    abnormal returns (AAR) and cumulative average abnormal returns (CAAR)
    and its significance tests.
    
    This implementation heavily relies on the work of MacKinlay [1]_.

    This class takes in input a list of single event studies (`eventstudy.Single`),
    aggregate them and gives access to aggregate statistics and tests.

    Note
    ----
    All single event studies must have the same specifications 
    (event, estimation and buffer windows).
    However, the model used for each event study can be different (if needed).

    References
    ----------

    .. [1] Mackinlay, A. (1997). “Event Studies in Economics and Finance”.
        In: Journal of Economic Literature 35.1, p. 13.
    """
    def __init__(
        self,
        sample: list,
        errors=None,
        remove_outliers: Callable=None,
        n: int=None,
        description: str=None
    ):
        """
        Low-level (complex) way of runing an aggregate of event studies.

        Parameters
        ----------
        sample : list
            List containing `eventstudy.Single` objects. 
            You can run independently each eventstudy, aggregate 
            them in a dictionary and compute their aggregate statistics.
        errors : list, optional
            A list containing errors encountered during the computation of single event studies, by default None.

        See also
        -------
        from_csv, from_list, from_pandas, from_excel

        Example
        -------

        Run an aggregate of event studies for Apple Inc. 10-K form releases. 
        We loop into a list of dates (in string format). 
        We first convert dates to a numpy.datetie64 format, 
        then run each event study, store them in an `events` list.
        Finally, we run the aggregate event study.
        
        1. Import packages:
        >>> import numpy as np
        >>> import datetime
        >>> import eventstudy as es

        2. import datas and initialize an empty list to store events:
        >>> es.Single.import_returns(cached=True)
        >>> dates = ['05/11/2018', '03/11/2017', '26/10/2016', 
        ...     '28/10/2015', '27/10/2014', '30/10/2013',
        ...     '31/10/2012', '26/10/2011', '27/10/2010']
        >>> events = list()

        3. Run each single event:
        >>> for date in dates:
        ...     formatted_date = np.datetime64(
        ...         datetime.datetime.strptime(date, '%d/%m/%Y')   
        ...     )
        ...     event = es.Single.market_model(
        ...         security_ticker = 'AAPL',
        ...         market_ticker = 'SPY',
        ...         event_date = formatted_date
        ...     )
        ...     events.append(event)

        4. Run the aggregate event study
        >>> agg = es.Multiple(events)
        """
        self.errors = errors
        self.__warn_errors()

        # retrieve common parameters from the first occurence in eventStudies:
        self.event_window = sample[0].event_window
        self.event_window_size = sample[0].event_window_size
        self.sample = sample
        self.total = len(sample)
        self.CAR = [event.CAR[-1] for event in sample]
        if remove_outliers:
            pred, scores = remove_outliers(self.CAR)
            # unset n when pred array length < n
            if n is not None and len(pred) < n:
                n = None
            mask = get_label_n(pred, scores, n=n) == 0
        else:
            mask = np.zeros(self.total, dtype=np.int64) == 0
        self.filtered = np.array(self.sample)[mask].tolist()
        self.filtered_total = len(self.filtered)
        self.description = description
        self.__compute()
        
    def __compute(self):
        
        weights = np.array([event.weight for event in self.filtered])
        total_weight = np.sum(weights)
        weights_ratios = np.expand_dims(weights / total_weight, axis=1)
        var_weights_ratios = np.expand_dims(weights / total_weight ** 2, axis=1)
        abnormal_returns = np.array([event.AR for event in self.filtered])
        var_abnormal_returns = np.array([event.var_AR for event in self.filtered])
        weighted_AR = abnormal_returns * weights_ratios
        weighted_var_AAR = var_abnormal_returns * var_weights_ratios
        self.AAR = np.sum(weighted_AR, axis=0)
        self.var_AAR = np.sum(weighted_var_AAR, axis=0)
        self.CAAR = np.cumsum(self.AAR)
        self.var_CAAR = [
            np.sum(self.var_AAR[:i]) for i in range(1, self.win_size + 1)
        ]

        self.tstat = self.CAAR / np.sqrt(self.var_CAAR)
        self.df = np.sum([event.df for event in self.filtered], axis=0)
        # see https://www.statology.org/t-distribution-python/
        self.pvalue = (1.0 - t.cdf(abs(self.tstat), self.df)) * 2

        self.CAR_dist = self.__compute_CAR_dist()

    def __weighted_mean(self, x, wts):
        return np.average(x, weights=wts, axis=0)

    def __weighted_variance(self, x, wts):
        return np.average((x - self.__weighted_mean(x, wts))**2, weights=wts, axis=0)
        
    def __weighted_kurtosis(self, x, wts):
        return (np.average((x - self.__weighted_mean(x, wts))**4, weights=wts, axis=0) /
            self.__weighted_variance(x, wts)**(2))

    def __weighted_skew(self, x, wts):
        return (np.average((x - self.__weighted_mean(x, wts))**3, weights=wts, axis=0) /
            self.__weighted_variance(x, wts)**(1.5))

    def __compute_CAR_dist(self):

        CAR = np.array([event.CAR for event in self.filtered])
        weights = np.array([event.weight for event in self.filtered])
        self.weights = weights
        self.CAR = CAR
        self.CAR_simple = np.exp(1) ** self.CAR - 1
        win_ct = np.where(np.array(CAR)> 0, 1, 0).sum(axis=0)
        total = len(CAR)
        CAR_dist = {
            "Win Ct": win_ct,
            "Total": total,
            "Win Ratio": win_ct / total,
            "Mean": self.__weighted_mean(CAR, weights),
            "Variance": self.__weighted_variance(CAR, weights),
            "Kurtosis": self.__weighted_kurtosis(CAR, weights),
            "Skewness": self.__weighted_skew(CAR, weights),
            "Min": np.min(CAR, axis=0),
            "Quantile 25%": np.quantile(CAR, q=0.25, axis=0),
            "Quantile 50%": np.quantile(CAR, q=0.5, axis=0),
            "Quantile 75%": np.quantile(CAR, q=0.75, axis=0),
            "Max": np.max(CAR, axis=0)
        }
        return CAR_dist

    def sign_test(self, sign="positive", confidence=0.9):
        """ Not implemented yet """
        # signtest
        # return nonParametricTest(self.CAR).signTest(sign, confidence)
        pass

    def rank_test(self, confidence):
        """ Not implemented yet """
        pass

    def results(self, asterisks: bool = True, decimals=4):
        """
        Give event study result in a table format.
        
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
            AAR and AAR's variance, CAAR and CAAR's variance, T-stat and P-value, 
            for each T in the event window.

        Note
        ----
        
        The function return a fully working pandas DataFrame.
        All pandas method can be used on it, especially exporting method (to_csv, to_excel,...)

        Example
        -------

        Get results of a market model event study on an 
        aggregate of events (Apple Inc. 10-K form releases) imported 
        from a csv, with specific number of decimal for each column:

        >>> events = es.Multiple.from_csv(
        ...     'AAPL_10K.csv',
        ...     es.Single.fama_french_3,
        ...     event_window = (-5,+5),
        ...     date_format = '%d/%m/%Y'
        ... )
        >>> events.results(decimals = [3,5,3,5,2,2])

        Note
        ----
        
        Significance level: \*\*\* at 99%, \*\* at 95%, \* at 90%
        """
        columns = {
            "AAR": self.AAR,
            "Std. E. AAR": np.sqrt(self.var_AAR),
            "CAAR": self.CAAR,
            "Std. E. CAAR": np.sqrt(self.var_CAAR),
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

    def plot(self, *, AAR=False, CI=True, confidence=0.95):
        """
        Plot the event study result.
        
        Parameters
        ----------
        AAR : bool, optional
            Add to the figure a bar plot of AAR, by default False
        CI : bool, optional
            Display the confidence interval, by default True
        confidence : float, optional
            Set the confidence level, by default 0.90
        
        Returns
        -------
        matplotlib.figure
            Plot of CAAR and AAR (if specified).

        Note
        ----
        The function return a fully working matplotlib function.
        You can extend the figure and apply new set-up with matplolib's method (e.g. savefig).
        
        Example
        -------

        Plot CAR (in blue) and AR (in black), with a confidence interval of 95% (in grey).

        >>> events = es.Multiple.from_csv(
        ...     'AAPL_10K.csv',
        ...     es.Single.fama_french_3,
        ...     event_window = (-5,+5),
        ...     date_format = '%d/%m/%Y'
        ... )
        >>> events.plot(AR = True, confidence = .95)

        .. image:: /_static/single_event_plot.png
        """
        return plot(
            time=range(self.event_window[0], self.event_window[1] + 1),
            CAR=self.CAAR,
            AR=self.AAR if AAR else None,
            CI=CI,
            var=self.var_CAAR,
            df=self.df,
            confidence=confidence,
        )

    def get_CAR_dist(self, decimals=3):
        """
        Give CARs' distribution descriptive statistics in a table format.
        
        Parameters
        ----------
        decimals : int or list, optional
            Round the value with the number of decimal specified, by default 3.
            `decimals` can either be an integer, in this case all value will be 
            round at the same decimals, or a list of 6 decimals, in this case each 
            columns will be round based on its respective number of decimal.

        Returns
        -------
        pandas.DataFrame
            CARs' descriptive statistics 

        Note
        ----
        
        The function return a fully working pandas DataFrame.
        All pandas method can be used on it, especially exporting method (to_csv, to_excel,...)

        Example
        -------

        Get CARs' descriptive statistics  of a market model event study on an
        aggregate of events (Apple Inc. 10-K release) imported 
        from a csv, with specific number of decimal for each column:

        >>> events = es.Multiple.from_csv(
        ...     'AAPL_10K.csv',
        ...     es.Single.fama_french_3,
        ...     event_window = (-5,+5),
        ...     date_format = '%d/%m/%Y'
        ... )
        >>> events.get_CAR_dist(decimals = 4)

        Note
        ----
        
        Significance level: \*\*\* at 99%, \*\* at 95%, \* at 90%
        """
        return to_table(
            self.CAR_dist, decimals=decimals, index_start=self.event_window[0]
        )

    def get_simple_CAR_dist(self, decimals=4):
        """
        Give CARs' distribution descriptive statistics in a table format.
        Converting the Log Retuns back to simple returns
        
        Parameters
        ----------
        decimals : int or list, optional
            Round the value with the number of decimal specified, by default 3.
            `decimals` can either be an integer, in this case all value will be 
            round at the same decimals, or a list of 6 decimals, in this case
            each columns will be round based on its respective number of
            decimal.
        Returns
        -------
        pandas.DataFrame
            CARs' descriptive statistics 
        Note
        ----
        
        The function return a fully working pandas DataFrame.
        All pandas method can be used on it, especially exporting method 
        (to_csv, to_excel,...)

        Example
        -------
        Get CARs' descriptive statistics  of a market model event study on an
        aggregate of events (Apple Inc. 10-K release) imported 
        from a csv, with specific number of decimal for each column:
        >>> events = es.MultipleEvents.from_csv(
        ...     'AAPL_10K.csv',
        ...     es.SingleEvent.fama_french_3,
        ...     event_window = (-5,+5)
        ... )
        >>> events.get_CAR_dist(decimals = 4)

        Note
        ----
        
        Significance level: \*\*\* at 99%, \*\* at 95%, \* at 90%
        """
        CAR = self.CAR_simple
        weights = self.weights
        win_ct = np.where(np.array(CAR)> 0, 1, 0).sum(axis=0)
        total = len(CAR)
        CAR_dist = {
            "Win Ct": win_ct,
            "Total": total,
            "Win Ratio": win_ct / total,
            "Mean": self.__weighted_mean(CAR, weights),
            "Variance": self.__weighted_variance(CAR, weights),
            "Kurtosis": self.__weighted_kurtosis(CAR, weights),
            "Skewness": self.__weighted_skew(CAR, weights),
            "Min": np.min(CAR, axis=0),
            "Quantile 25%": np.quantile(CAR, q=0.25, axis=0),
            "Quantile 50%": np.quantile(CAR, q=0.5, axis=0),
            "Quantile 75%": np.quantile(CAR, q=0.75, axis=0),
            "Max": np.max(CAR, axis=0)
        }
        self.CAR_dist_simple = CAR_dist
        return to_table(
            CAR_dist, decimals=decimals, index_start=self.event_window[0]
        )

    @classmethod
    def from_list(
        cls,
        event_list: list,
        event_model,
        event_window: tuple = (-5, +5),
        est_size: int = 252,
        buffer_size: int = 21,
        weight: int = 1,
        *,
        remove_outliers: Callable = None,
        n: int = None,
        ignore_errors: bool = True
    ):
        """
        Compute an aggregate of event studies from a list containing each
        event's parameters.
        
        Parameters
        ----------
        event_list : list
            List containing dictionaries specifing each event's parameters 
            (see example for more details).
        event_model
            Function returning an eventstudies.SingleEvent class instance.
            For example, eventstudies.SingleEvent.MarketModel() 
            (a custom functions can be created).
        event_window : tuple, optional
            Event window specification (T2,T3), by default (-5, +5).
            A tuple of two integers, representing the start and the end of 
            the event window. Classically, the event-window starts before 
            the event and ends after the event. For example, `event_window 
            = (-2,+20)` means that the event-period starts 2 periods before
            the event and ends 20 periods after.
        est_size : int, optional
            Size of the estimation for the modelisation of returns 
            [T0,T1], by default 252
        buffer_size : int, optional
            Size of the buffer window [T1,T2], by default 21
        weight : int, optional
            Weight to be applied to the returns in the MultipleEvents Object
        remove_outliers : function, optional, default None
            one of the following outlier models from eventstudies.outliers 
            'from eventstudies.outliers import ECOD, MAD, SOS, KNN, LOF, IForest'
            This will remove outliers based upon the following outlier 
            detection models from the pyod module.
        n : int, optional - default None
            If remove_outliers and n is set, outlier detection will remove the
            top n outliers from the data set.  If remove_outliers is set and
             n is not, it will remove all outliers from the data set.
        ignore_errors : bool, optional
            If true, errors during the computation of single event studies will
            be ignored. In this case, these events will be removed from the
            computation.  However, a warning message will be displayed after
            the computation to warn for errors.  Errors can also be accessed
            using `print(eventstudy.MultipleEvents.error_report())`. 
            If false, the computation will be stopped by any error encounter
            during the computation of single event studies, by default True
            
        See also
        --------
        
        from_csv, from_pandas, from_excel
        Example
        -------
        >>> list = [
        ...     {'event_date': np.datetime64("2018-11-05"), 'security_ticker': 'AAPL'},
        ...     {'event_date': np.datetime64("2017-11-03"), 'security_ticker': 'AAPL'},
        ...     {'event_date': np.datetime64("2016-10-26"), 'security_ticker': 'AAPL'},
        ...     {'event_date': np.datetime64("2015-10-28"), 'security_ticker': 'AAPL'},
        ... ]
        >>> agg = eventstudy.MultipleEvents.from_list(
        ...     event_list = list,
        ...     event_model = eventstudies.SingleEvent.fama_french_3,
        ...     event_window = (-5,+10),
        ... ) 
        """

        # event_list = [
        #   {'event_date': np.datetime64, models_data},
        #   {'event_date': np.datetime64, models_data}
        # ]
        sample = list()
        errors = list()
        # parsing only the keys that are needed allows passing of a dict with 
        # more than just the values below without breaking the models with 
        # unrecognized keys. This is useful for grouping outside the models.
        event_param_keys = ('security_ticker', 'event_date', 'mkt_idx')
        for event_params in event_list:
            try:
                ev_params = {}
                for k in event_param_keys:
                    if k in event_params:   
                        ev_params[k] = event_params[k]
                event = event_model(
                    **ev_params,
                    event_window=event_window,
                    est_size=est_size,
                    buffer_size=buffer_size,
                    weight=weight
                )
            except (DateMissingError, DataMissingError, ColumnMissingError) as e:
                if ignore_errors:
                    event_params["error_type"] = e.__class__.__name__
                    event_params["error_msg"] = e.helper
                    errors.append(event_params)
                else:
                    raise e
            else:
                sample.append(event)

        return cls(sample, errors, remove_outliers, n)

    @classmethod
    def from_csv(
        cls,
        path,
        event_model,
        event_window: tuple = (-5, +5),
        est_size: int = 252,
        buffer_size: int = 21,
        weight: int = 1,
        *,
        remove_outliers: Callable = None,
        n: int = None,
        ignore_errors: bool = True,
        sep='|'
    ):
        """
        Compute an aggregate of event studies from a csv file containing each
        event's parameters.
        
        Parameters
        ----------
        path : str
            Path to the csv file containing events' parameters.
            The first line must contains the name of each parameter needed to
            compute the event_model. All values must be separated by a comma.
        event_model
            Function returning an eventstudies.SingleEvent class instance.
            For example, eventstudies.SingleEvent.MarketModel() (a custom 
            functions can be created).
        event_window : tuple, optional
            Event window specification (T2,T3), by default (-5, +5).
            A tuple of two integers, representing the start and the end of the
            event window. Classically, the event-window starts before the event
            and ends after the event. For example, `event_window = (-2,+20)` 
            means that the event-period starts 2 periods before the event and
            ends 20 periods after.
        est_size : int, optional
            Size of the estimation for the modelisation of returns [T0,T1],
            by default 252
        buffer_size : int, optional
            Size of the buffer window [T1,T2], by default 21
        weight : int, optional
            Weight to be applied to the returns in the MultipleEvents Object
        remove_outliers : function, optional, default None
            one of the following outlier models from eventstudies.outliers 
            'from eventstudies.outliers import ECOD, MAD, SOS, KNN, LOF, IForest'
            This will remove outliers based upon the following outlier 
            detection models from the pyod module.
        n : int, optional - default None
            If remove_outliers and n is set, outlier detection will remove the
            top n outliers from the data set.  If remove_outliers is set and n 
            is not, it will remove all outliers from the data set.
        ignore_errors : bool, optional
            If true, errors during the computation of single event studies will
            be ignored. In this case, these events will be removed from the 
            computation.  However, a warning message will be displayed after 
            the computation to warn for errors.  Errors can also be accessed 
            using `print(eventstudy.MultipleEvents.error_report())`.
            If false, the computation will be stopped by any error encounter 
            during the computation of single event studies, by default True
            
        See also
        --------
        
        from_list, from_pandas, from_excel
        Example
        -------
        >>> agg = eventstudy.MultipleEvents.from_csv(
        ...     path = 'events.csv',
        ...     event_model = eventstudies.SingleEvent.MarketModel,
        ...     event_window = (-5,+10),
        ...     date_format = "%d/%m/%Y"
        ... ) 
        """

        event_list = pd.read_csv(
            path,
            parse_dates=["event_date"],
            sep=sep,
            header=0
        ).to_dict('records')

        return cls.from_list(
            event_list,
            event_model,
            event_window,
            est_size,
            buffer_size,
            weight=weight,
            remove_outliers=remove_outliers,
            n=n,
            ignore_errors=ignore_errors
        )
    
    @classmethod
    def from_excel(
        cls,
        path,
        event_model,
        event_window: tuple = (-5, +5),
        est_size: int = 252,
        buffer_size: int = 21,
        weight: int = 1,
        *,
        remove_outliers: Callable = None,
        n: int = None,
        ignore_errors: bool = True,
        sheet_name=0
    ):
        """
        Compute an aggregate of event studies from a csv file containing each
        event's parameters.
        
        Parameters
        ----------
        path : str
            Path to the csv file containing events' parameters.
            The first line must contains the name of each parameter needed to
            compute the event_model. All values must be separated by a comma.
        event_model
            Function returning an eventstudies.SingleEvent class instance.
            For example, eventstudies.SingleEvent.MarketModel() (a custom 
            functions can be created).
        event_window : tuple, optional
            Event window specification (T2,T3), by default (-5, +5).
            A tuple of two integers, representing the start and the end of the
            event window. Classically, the event-window starts before the event
            and ends after the event. For example, `event_window = (-2,+20)` 
            means that the event-period starts 2 periods before the event and
            ends 20 periods after.
        est_size : int, optional
            Size of the estimation for the modelisation of returns [T0,T1],
            by default 252
        buffer_size : int, optional
            Size of the buffer window [T1,T2], by default 21
        weight : int, optional
            Weight to be applied to the returns in the MultipleEvents Object
        remove_outliers : function, optional, default None
            one of the following outlier models from eventstudies.outliers 
            'from eventstudies.outliers import ECOD, MAD, SOS, KNN, LOF, IForest'
            This will remove outliers based upon the following outlier 
            detection models from the pyod module.
        n : int, optional - default None
            If remove_outliers and n is set, outlier detection will remove the
            top n outliers from the data set.  If remove_outliers is set and n 
            is not, it will remove all outliers from the data set.
        ignore_errors : bool, optional
            If true, errors during the computation of single event studies will
            be ignored. In this case, these events will be removed from the 
            computation.  However, a warning message will be displayed after 
            the computation to warn for errors.  Errors can also be accessed 
            using `print(eventstudy.MultipleEvents.error_report())`.
            If false, the computation will be stopped by any error encounter 
            during the computation of single event studies, by default True
            
        See also
        --------
        
        from_list, from_pandas, from_excel
        Example
        -------
        >>> agg = eventstudy.MultipleEvents.from_excel(
        ...     path = 'events.xlsx',
        ...     event_model = eventstudies.SingleEvent.MarketModel,
        ...     event_window = (-5,+10),
        ...     date_format = "%d/%m/%Y"
        ... ) 
        """

        event_list = pd.read_excel(
            path,
            parse_dates=["event_date"],
            header=0,
            sheet_name=sheet_name).to_dict('records')

        return cls.from_list(
            event_list,
            event_model,
            event_window,
            est_size,
            buffer_size,
            weight=weight,
            remove_outliers=remove_outliers,
            n=n,
            ignore_errors=ignore_errors
        )
    
    @classmethod
    def from_pandas(
        cls,
        df,
        event_model,
        event_window: tuple = (-5, +5),
        est_size: int = 252,
        buffer_size: int = 21,
        weight: int = 1,
        *,
        remove_outliers: Callable=None,
        n: int = None,
        ignore_errors: bool = True    
    ):
        """

        Compute an aggregate of event studies from a csv file containing each
        event's parameters.
        
        Parameters
        ----------
        path : str
            Path to the csv file containing events' parameters.
            The first line must contains the name of each parameter needed to
            compute the event_model. All values must be separated by a comma.
        event_model
            Function returning an eventstudies.SingleEvent class instance.
            For example, eventstudies.SingleEvent.MarketModel() (a custom 
            functions can be created).
        event_window : tuple, optional
            Event window specification (T2,T3), by default (-5, +5).
            A tuple of two integers, representing the start and the end of the
            event window. Classically, the event-window starts before the event
            and ends after the event. For example, `event_window = (-2,+20)` 
            means that the event-period starts 2 periods before the event and
            ends 20 periods after.
        est_size : int, optional
            Size of the estimation for the modelisation of returns [T0,T1],
            by default 252
        buffer_size : int, optional
            Size of the buffer window [T1,T2], by default 21
        weight : int, optional
            Weight to be applied to the returns in the MultipleEvents Object
        remove_outliers : function, optional, default None
            one of the following outlier models from eventstudies.outliers 
            'from eventstudies.outliers import ECOD, MAD, SOS, KNN, LOF, IForest'
            This will remove outliers based upon the following outlier 
            detection models from the pyod module.
        n : int, optional - default None
            If remove_outliers and n is set, outlier detection will remove the
            top n outliers from the data set.  If remove_outliers is set and n 
            is not, it will remove all outliers from the data set.
        ignore_errors : bool, optional
            If true, errors during the computation of single event studies will
            be ignored. In this case, these events will be removed from the 
            computation.  However, a warning message will be displayed after 
            the computation to warn for errors.  Errors can also be accessed 
            using `print(eventstudy.MultipleEvents.error_report())`.
            If false, the computation will be stopped by any error encounter 
            during the computation of single event studies, by default True
            
        See also
        --------
        
        from_list, from_pandas, from_excel
        Example
        -------
        >>> agg = eventstudy.MultipleEvents.from_pandas(
        ...     df,
        ...     event_model = eventstudies.SingleEvent.MarketModel,
        ...     event_window = (-5,+10)
        ... )
        """

        event_list = df.to_dict('records')

        return cls.from_list(
            event_list,
            event_model,
            event_window,
            est_size,
            buffer_size,
            weight=weight,
            remove_outliers=remove_outliers,
            n=n,
            ignore_errors=ignore_errors
        )

    def __warn_errors(self):
        if self.errors is not None:
            nb = len(self.errors)
            if nb > 0:
                if nb > 1:
                    msg = " ".join([
                        f" {str(nb)} events have not been processed due",
                        "to data issues."])
                else:
                    msg = "1 event has not been processed due to data issues."

                msg += " ".join([
                    "\nTips: Get more details on errors by calling ",
                    " MultipleEvents.error_report() method",
                    " or by exploring MultipleEvents.errors class variable."])
                log.warning(msg)

    def error_report(self):
        """
        Return a report of errors faced during the computation of event studies.
        Example
        -------
        
        >>> agg = eventstudy.MultipleEvents.from_csv(
        ...     path = 'events.csv',
        ...     event_model = eventstudies.SingleEvent.MarketModel
        ... )
        >>> print(agg.error_report())
        """

        if self.errors is not None and len(self.errors) > 0:
            nb = (
                f"One error"
                if len(self.errors) == 1
                else f"{str(len(self.errors))} errors"
            )
            report = f"""Error Report
============
{nb} due to data unavailability.
The respective events was not processed and thus removed from the sample.
It does not affect the computation of other events.
Help 1: Check if the company was quoted at this date, 
Help 2: For event study modelised used Fama-French models,
        check if the Fama-French dataset imported is up-to-date.
Tips:   Re-import all parameters and re-run the event study analysis.
Details
=======
(You can find more details on errors in the documentation.)
"""

            table = list()
            lengths = {"error_type": 5, "date": 5, "error_msg": 0, "parameters": 11}

            for error in self.errors:
                # to preserve the instance errors variable from being modified
                error = error.copy()
                cells = {
                    "error_type": str(error.pop("error_type")),
                    "date": str(error.pop("event_date")),
                    "error_msg": str(error.pop("error_msg")),
                    "parameters": "; ".join(
                        [f"{str(key)}: {str(value)}" for key, value in error.items()]
                    ),
                }
                table.append(cells)

                for key, cell in cells.items():
                    if len(cell) > lengths[key]:
                        lengths[key] = len(cell)

            header = (
                "Error".ljust(lengths["error_type"])
                + " Date".ljust(lengths["date"])
                + "  Parameters".ljust(lengths["parameters"])
                + "\n"
            )
            mid_rule = (
                "-" * lengths["error_type"]
                + " "
                + "-" * lengths["date"]
                + " "
                + "-" * lengths["parameters"]
            )
            table_str = ""
            for cells in table:
                table_str += (
                    "\n"
                    + f"{cells['error_type']}".ljust(lengths["error_type"])
                    + " "
                    + f"{cells['date']}".ljust(lengths["date"])
                    + " "
                    + f"{cells['parameters']}".ljust(lengths["parameters"])
                    + "\nDescription: "
                    + cells["error_msg"]
                    + "\n"
                )

            report += header + mid_rule + table_str + mid_rule

            return report

        else:
            return "No error."