import os
import warnings
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Iterable, List
from schema.data_schema import ForecastingSchema
from sklearn.exceptions import NotFittedError
from mlforecast import MLForecast
from sklearn.neural_network import MLPRegressor
from mlforecast.target_transforms import LocalMinMaxScaler

warnings.filterwarnings("ignore")


PREDICTOR_FILE_NAME = "predictor.joblib"


class Forecaster:
    """A wrapper class for the MLP Forecaster.

    This class provides a consistent interface that can be used with other
    Forecaster models.
    """

    model_name = "MLP Forecaster"

    def __init__(
        self,
        data_schema: ForecastingSchema,
        history_forecast_ratio: int = None,
        lags_forecast_ratio: int = None,
        lags: Optional[Iterable] = None,
        hidden_layer_sizes: List[int] = [100],
        activation: str = "relu",
        solver: str = "adam",
        learning_rate: str = "constant",
        alpha: float = 0.0,
        use_exogenous: bool = True,
        random_state: int = 0,
        **kwargs,
    ):
        """Construct a new MLP Forecaster

        Args:

            data_schema (ForecastingSchema):
                Schema of training data.

            history_forecast_ratio (int):
                Sets the history length depending on the forecast horizon.
                For example, if the forecast horizon is 20 and the history_forecast_ratio is 10,
                history length will be 20*10 = 200 samples.

            lags_forecast_ratio (int):
                Sets the lags parameters depending on the forecast horizon.
                lags = forecast horizon * lags_forecast_ratio
                This parameters overides lags parameter and uses the most recent values as lags.

            lags (Optional[Iterable]): Lags of the target to use as features.

            hidden_layer_sizes (List[int]): The ith element represents the number of neurons in the ith hidden layer.

            activation (str):
                {'identity', 'logistic', 'tanh', 'relu'}
                Activation function for the hidden layer.

                - 'identity', no-op activation, useful to implement linear bottleneck, returns f(x) = x
                - 'logistic', the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).
                - 'tanh', the hyperbolic tan function, returns f(x) = tanh(x).
                - 'relu', the rectified linear unit function, returns f(x) = max(0, x)


            solver (str):
                {'lbfgs', 'sgd', 'adam'}
                The solver for weight optimization.

                - 'lbfgs' is an optimizer in the family of quasi-Newton methods.
                - 'sgd' refers to stochastic gradient descent.
                - 'adam' refers to a stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba

                Note: The default solver 'adam' works pretty well on relatively large datasets (with thousands of training samples or more) in terms of both training time and validation score. For small datasets, however, 'lbfgs' can converge faster and perform better.

            learning_rate (str):
                {'constant', 'invscaling', 'adaptive'}, default='constant'
                Learning rate schedule for weight updates.

                - 'constant' is a constant learning rate given by 'learning_rate_init'.
                - 'invscaling' gradually decreases the learning rate learning_rate_ at each time step 't' using an inverse scaling exponent of 'power_t'. effective_learning_rate = learning_rate_init / pow(t, power_t)
                - 'adaptive' keeps the learning rate constant to 'learning_rate_init' as long as training loss keeps decreasing. Each time two consecutive epochs fail to decrease training loss by at least tol, or fail to increase validation score by at least tol if 'early_stopping' is on, the current learning rate is divided by 5.

            alpha (float):
                Strength of the L2 regularization term. The L2 regularization term is divided by the sample size when added to the loss.

            use_exogenous (bool): If true, uses covariates in training.

            random_state (int): Sets the underlying random seed at model initialization time.
        """
        self.data_schema = data_schema
        self.lags = lags
        self.use_exogenous = use_exogenous
        self.random_state = random_state
        self._is_trained = False
        self.kwargs = kwargs
        self.history_length = None

        hidden_layer_sizes = tuple(hidden_layer_sizes)

        if history_forecast_ratio:
            self.history_length = (
                self.data_schema.forecast_length * history_forecast_ratio
            )

        if lags_forecast_ratio:
            lags = lags_forecast_ratio * self.data_schema.forecast_length
            self.lags = [i for i in range(1, lags + 1)]

        models = [
            MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                learning_rate=learning_rate,
                activation=activation,
                solver=solver,
                alpha=alpha,
                **kwargs,
            )
        ]

        self.model = MLForecast(
            models=models,
            freq=self.map_frequency(data_schema.frequency),
            lags=self.lags,
            target_transforms=[LocalMinMaxScaler()],
        )

    def map_frequency(self, frequency: str) -> str:
        """
        Maps the frequency in the data schema to the frequency expected by mlforecast.

        Args:
            frequency (str): The frequency from the schema.

        Returns (str): The mapped frequency.
        """
        if self.data_schema.time_col_dtype == "INT":
            return 1

        frequency = frequency.lower()
        frequency = frequency.split("frequency.")[1]
        if frequency == "yearly":
            return "Y"
        if frequency == "quarterly":
            return "Q"
        if frequency == "monthly":
            return "M"
        if frequency == "weekly":
            return "W"
        if frequency == "daily":
            return "D"
        if frequency == "hourly":
            return "H"
        if frequency == "minutely":
            return "min"
        if frequency == "secondly":
            return "S"
        else:
            return 1

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares the training data by dropping past covariates, converting the index to datetime if available
        and drops or keeps other covariates depending on use_exogenous.

            Args:
                data (pd.DataFrame): The training data.
        """
        data.drop(columns=self.data_schema.past_covariates, inplace=True)

        if self.data_schema.time_col_dtype in ["DATE", "DATETIME"]:
            data[self.data_schema.time_col] = pd.to_datetime(
                data[self.data_schema.time_col]
            )

        if not self.use_exogenous:
            if self.data_schema.future_covariates:
                data.drop(columns=self.data_schema.future_covariates, inplace=True)

            if self.data_schema.static_covariates:
                data.drop(columns=self.data_schema.static_covariates, inplace=True)

        return data

    def fit(
        self,
        history: pd.DataFrame,
    ) -> None:
        """Fit the Forecaster to the training data.

        Args:
            history (pandas.DataFrame): The features of the training data.

        """
        np.random.seed(self.random_state)

        if self.use_exogenous and len(self.data_schema.static_covariates) > 0:
            static_features = self.data_schema.static_covariates

        else:
            static_features = []

        history = self.prepare_data(history)

        self.model.fit(
            df=history,
            time_col=self.data_schema.time_col,
            id_col=self.data_schema.id_col,
            target_col=self.data_schema.target,
            static_features=static_features,
        )
        self._is_trained = True

    def predict(
        self, test_data: pd.DataFrame, prediction_col_name: str
    ) -> pd.DataFrame:
        """Make the forecast of given length.

        Args:
            test_data (pd.DataFrame): Given test input for forecasting.
            prediction_col_name (str): Name to give to prediction column.
        Returns:
            pd.DataFrame: The prediction dataframe.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")

        if self.use_exogenous and self.data_schema.future_covariates:
            future_df = self.model.make_future_dataframe(
                self.data_schema.forecast_length
            )
            future_df[self.data_schema.future_covariates] = test_data[
                self.data_schema.future_covariates
            ]
        else:
            future_df = None

        forecast = self.model.predict(self.data_schema.forecast_length, X_df=future_df)
        forecast[prediction_col_name] = forecast.drop(
            columns=[self.data_schema.time_col, self.data_schema.id_col]
        ).mean(axis=1)
        forecast[self.data_schema.time_col] = test_data[self.data_schema.time_col]
        return forecast

    def save(self, model_dir_path: str) -> None:
        """Save the Forecaster to disk.

        Args:
            model_dir_path (str): Dir path to which to save the model.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        joblib.dump(self, os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "Forecaster":
        """Load the Forecaster from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            Forecaster: A new instance of the loaded Forecaster.
        """
        model = joblib.load(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))
        return model

    def __str__(self):
        # sort params alphabetically for unit test to run successfully
        return f"Model name: {self.model_name}"


def train_predictor_model(
    history: pd.DataFrame,
    data_schema: ForecastingSchema,
    hyperparameters: dict,
) -> Forecaster:
    """
    Instantiate and train the predictor model.

    Args:
        history (pd.DataFrame): The training data inputs.
        data_schema (ForecastingSchema): Schema of the training data.
        hyperparameters (dict): Hyperparameters for the Forecaster.

    Returns:
        'Forecaster': The Forecaster model
    """

    model = Forecaster(
        data_schema=data_schema,
        **hyperparameters,
    )
    model.fit(
        history=history,
    )
    return model


def predict_with_model(
    model: Forecaster, test_data: pd.DataFrame, prediction_col_name: str
) -> pd.DataFrame:
    """
    Make forecast.

    Args:
        model (Forecaster): The Forecaster model.
        test_data (pd.DataFrame): The test input data for forecasting.
        prediction_col_name (int): Name to give to prediction column.

    Returns:
        pd.DataFrame: The forecast.
    """
    return model.predict(test_data, prediction_col_name)


def save_predictor_model(model: Forecaster, predictor_dir_path: str) -> None:
    """
    Save the Forecaster model to disk.

    Args:
        model (Forecaster): The Forecaster model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> Forecaster:
    """
    Load the Forecaster model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        Forecaster: A new instance of the loaded Forecaster model.
    """
    return Forecaster.load(predictor_dir_path)


def evaluate_predictor_model(
    model: Forecaster, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    Evaluate the Forecaster model and return the accuracy.

    Args:
        model (Forecaster): The Forecaster model.
        x_test (pd.DataFrame): The features of the test data.
        y_test (pd.Series): The labels of the test data.

    Returns:
        float: The accuracy of the Forecaster model.
    """
    return model.evaluate(x_test, y_test)
