import numpy as np
import statsmodels.api as sm


class Model:

    """
    The Model Class holds various methods to run the various returns models
    It takes the data needed to compute the model as parameters
    and the following parameters:
        estimation_size: int,
        event_window_size: int

    Each method returns:
        an array of residuals of the event window
        the degrees of freedom
        an array of the variance of the residuals
        the model used
    """

    def __init__(
        self, estimation_size: int, event_window_size: int
    ):
        self.estimation_size = estimation_size
        self.event_window_size = event_window_size

    def OLS(self, X, Y):
        X = sm.add_constant(X)  # add an intercept
        
        # trim leading 0 returns from the dataset prior to regression
        l = len(Y)
        Yt = np.trim_zeros(Y)
        lt = len(Yt)
        lz = l - lt
        Y = Yt
        X = X[lz:, :]
        if Y[: self.estimation_size].shape[0] > 0:
            model = sm.OLS(Y[: self.estimation_size], X[: self.estimation_size]).fit()
            residuals = np.array(Y) - model.predict(X)
        else:
            model = None
            residuals = np.zeros(np.max([l, self.event_window_size]))
        
        df = self.estimation_size - 1
        var = np.var(residuals[: self.estimation_size])
        # if residual length is less than win_size after trimming
        # we need to pad the front again so it returns the length 
        # expected
        if len(residuals) < self.event_window_size:
            pad_size = self.event_window_size - len(residuals)
            pad = np.zeros(pad_size)
            residuals = np.concatenate([pad, residuals])
        return residuals[-self.event_window_size :], df, var, model


def market_model(
    security_returns,
    market_returns,
    *,  # Named arguments only
    estimation_size: int,
    event_window_size: int,
    **kwargs
):
    X = np.array(market_returns)
    Y = np.array(security_returns)
    residuals, df, var_res, model = Model(
        estimation_size,
        event_window_size).OLS(
            X,
            Y)
    var = [var_res] * event_window_size
    return residuals, df, var, model


def fama_french_3(
    security_returns,
    Mkt_RF,
    SMB,
    HML,
    RF,
    *,  # Named arguments only
    estimation_size: int,
    event_window_size: int,
    **kwargs
):

    RF = np.array(RF)
    Mkt_RF = np.array(Mkt_RF)
    security_returns = np.array(security_returns)
    X = np.column_stack((Mkt_RF, SMB, HML))
    Y = np.array(security_returns) - np.array(RF)
    residuals, df, var_res, model = Model(
        estimation_size,
        event_window_size).OLS(
        X, Y)
    var = [var_res] * event_window_size
    return residuals, df, var, model
   
   
def mean_adjusted_model(
    security_returns,
    *,  # Named arguments only
    estimation_size: int,
    event_window_size: int,
    **kwargs
):
    mean = np.mean(security_returns[:estimation_size])
    residuals = np.array(security_returns) - mean
    df = estimation_size - 1
    var = [np.var(residuals)] * event_window_size
    return residuals[-event_window_size:], df, var, mean


def market_adjusted_model(
    security_returns,
    Mkt_RF,
    *,  # Named arguments only
    estimation_size: int,
    event_window_size: int,
    **kwargs
):
    X = np.array(security_returns)
    Y = np.array(Mkt_RF)
    residuals = Y - X
    df = estimation_size - 1
    var = [np.var(residuals)] * event_window_size
    return residuals[-event_window_size:], df, var, X[-event_window_size:]


def ordinary_returns_model(
    security_returns,
    *,  # Named arguments only
    estimation_size: int,
    event_window_size: int,
    **kwargs
):
    X = np.array(security_returns)
    Y = np.array([0] * len(X))
    residuals = X - Y
    df = estimation_size - 1
    var = [np.var(residuals)] * event_window_size
    return residuals[-event_window_size:], df, var, X[-event_window_size:]


def fama_french_5(
    security_returns,
    Mkt_RF,
    SMB,
    HML,
    RMW,
    CMA,
    RF,
    *,  # Named arguments only
    estimation_size: int,
    event_window_size: int,
    **kwargs
):

    RF = np.array(RF)
    Mkt_RF = np.array(Mkt_RF)
    security_returns = np.array(security_returns)
    X = np.column_stack((Mkt_RF, SMB, HML, RMW, CMA))
    Y = np.array(security_returns) - np.array(RF)
    residuals, df, var_res, model = Model(
        estimation_size,
        event_window_size).OLS(
            X, Y)
    var = [var_res] * event_window_size
    return residuals, df, var, model

def carhart(
    security_returns,
    Mkt_RF,
    SMB,
    HML,
    RF,
    MOM,
    *,  # Named arguments only
    estimation_size: int,
    event_window_size: int,
    **kwargs
):
    RF = np.array(RF)
    Mkt_RF = np.array(Mkt_RF)
    daily_ret = np.array(security_returns)
    X = np.column_stack((Mkt_RF, SMB, HML, MOM))
    Y = np.array(security_returns) - np.array(RF)
    residuals, df, var_res, model = Model(
        estimation_size,
        event_window_size).OLS(
            X, Y)
    var = [var_res] * event_window_size
    return residuals, df, var, model 