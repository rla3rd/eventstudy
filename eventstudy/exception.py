class CustomException(BaseException):
    pass


class ReturnsCacheEmptyError(CustomException):
    def __init__(self):
        super(ReturnsCacheEmptyError, self).__init__()
        self.msg = "Returns cache is empty, populating..."
    def __str__(self):
        return self.msg
    

class FutureDateError(CustomException):
    def __init__(self, date=None):
        super(FutureDateError, self).__init__()
        self.msg = f"Date {date} is in the future."
    def __str__(self):
        return self.msg


class EventFormatError(CustomException):
    def __init__(self):
        super(EventFormatError, self).__init__()
        self.msg = f"Event is not of type dict"
    def __str__(self):
        return self.msg
    

class EventKeyError(CustomException):
    def __init__(self, key=None):
        super(EventKeyError, self).__init__()
        self.msg = f"Event {key} is not in dict"
    def __str__(self):
        return self.msg


class ParameterMissingError(CustomException):
    def __init__(self, param_name=None):
        super(ParameterMissingError, self).__init__()
        if param_name == "returns":
            self.helper = "Returns are missing."
            self.msg = (
                self.helper + "\nTips: Import returns first using: "
                "EventStudy.Single.import_returns() or EventStudy.Single.import_returns_from_API()"
            )
        elif param_name == "FamaFrench":
            self.helper = "Fama-French factors are missing."
            self.msg = (
                self.helper + "\nTips: Import Fama-French factors first using: "
                "EventStudy.Single.import_FamaFrench() or EventStudy.Single.import_FamaFrench_from_API()"
            )
        else:
            if param_name:
                self.msg = f"One parameter is missing: {str(param_name)}."
            else:
                self.msg = "One parameter is missing."
            self.helper = self.msg

    def __str__(self):
        return self.msg


class ColumnMissingError(CustomException):
    def __init__(self, param_name=None, column=None):
        super(ColumnMissingError, self).__init__()
        if param_name == "returns":
            self.helper = f"{column} ticker is not available in returns parameter."
            self.msg = (
                self.helper + "\nTips: Re-import returns with this ticker using: "
                "EventStudy.Single.import_returns() or EventStudy.Single.import_returns_from_API()"
            )
        elif param_name == "FamaFrench":
            self.helper = (
                f"{column} factor is not available in Fama-French factors parameter."
            )
            self.msg = (
                self.helper + "\nTips: Re-import Fama-French factors using: "
                "eventstudies.SingleEvent.import_FamaFrench()"
            )
        else:
            if param_name:
                self.msg = f"One column is missing in parameter: {str(param_name)}."
            else:
                self.msg = "One parameter is missing."
            self.helper = self.msg

    def __str__(self):
        return self.msg


class DateMissingError(CustomException):
    def __init__(self, date=None, param_name=None):
        super(DateMissingError, self).__init__()

        if date and param_name:
            self.helper = (
                f"Date ({str(date)}) is missing in parameter: {str(param_name)}."
            )
        elif date:
            self.helper = f"Date ({str(date)}) is missing in parameters."
        elif param_name:
            self.helper = f"A date is missing in parameter: {str(param_name)}."
        else:
            self.helper = f"A date is missing in parameters."

        self.msg = (
            self.helper
            + "\nTips: Check if class parameters are up-to-date and contain all needed value."
        )

    def __str__(self):
        return self.msg


class DataMissingError(CustomException):
    def __init__(
        self, param_name=None, column=None, actual_size=None, expected_size=None, extra_msg=""
    ):
        super(DataMissingError, self).__init__()

        if param_name and column:
            self.helper = (
                f"Some data are missing for ({str(column)}) in '{str(param_name)}''."
            )
        elif param_name:
            self.helper = f"Some data are missing in '{str(param_name)}''."
        elif column:
            self.helper = f"Some data are missing for ({str(column)}) in parameters."
        else:
            self.helper = f"Some data are missing in parameters."

        if actual_size and expected_size:
            self.helper += f"\n{str(actual_size)} data retrieved over {str(expected_size)} expected."

        if extra_msg != "":
            self.helper += f"\n{extra_msg}"

        self.msg = (
            self.helper
            + "\nTips: Check if parameters are up-to-date and contain all needed value."
        )

    def __str__(self):
        return self.msg
