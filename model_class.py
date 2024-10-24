import glob
import os
import pickle

import numpy as np
import pandas as pd


class F1PredictionPipeline:
    def __init__(self):
        """
        Initialize the F1 Prediction Pipeline with model and encoder paths.
        Models and encoders are expected to be in 'models' and 'encoders' directories respectively.
        """
        self.feature_data = None
        self._processed_weather = None
        self._processed_lap_data = None
        self.model_paths = {
            'n_stint': os.path.join('models', 'n_stint_model.pkl'),
            'compound': os.path.join('models', 'compound_model.pkl'),
            'pit_lap': os.path.join('models', 'pit_lap_model.pkl'),
            'lap_time': os.path.join('models', 'lap_time_model.pkl')
        }

        self.scaler_paths = {
            'label_encoders_n_stint': os.path.join('encoders', 'label_encoders_n_stint.pkl'),
            'label_encoders_comp': os.path.join('encoders', 'label_encoders_comp.pkl'),
            'label_encoders_lap': os.path.join('encoders', 'label_encoders_lap.pkl'),
        }

        self.models = {}
        self.scalers = {}

        self.lap_data = pd.DataFrame()
        self.weather_data = pd.DataFrame()
        self.result_data = pd.DataFrame()

        self._load_models()
        self._load_encoders_scalers()

    def _load_models(self):
        """Load all prediction models from pickle files."""
        self.models['n_stint'] = pickle.load(open(self.model_paths['n_stint'], 'rb'))
        self.models['compound'] = pickle.load(open(self.model_paths['compound'], 'rb'))
        self.models['pit_lap'] = pickle.load(open(self.model_paths['pit_lap'], 'rb'))
        self.models['lap_time'] = pickle.load(open(self.model_paths['lap_time'], 'rb'))

    def _load_encoders_scalers(self):
        """Load all label encoders from pickle files."""
        self.scalers['label_encoders_n_stint'] = pickle.load(open(self.scaler_paths['label_encoders_n_stint'], 'rb'))
        self.scalers['label_encoders_comp'] = pickle.load(open(self.scaler_paths['label_encoders_comp'], 'rb'))
        self.scalers['label_encoders_lap'] = pickle.load(open(self.scaler_paths['label_encoders_lap'], 'rb'))

    @staticmethod
    def convert_to_seconds(df: pd.DataFrame, columns=None):
        """
        Convert time-based columns from string format (MM:SS.mmm) to total seconds.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing time data
        columns (list, optional): List of column names to convert. Defaults to
            ['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time']

        Returns:
        pd.DataFrame: DataFrame with specified columns converted to seconds (float)
        """
        if columns is None:
            columns = ['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time']
        for col in columns:
            if not df[col].dtype == 'float64':
                df[col] = pd.to_timedelta(df[col]).dt.total_seconds()
        return df

    def load_data(self, data_path=None):
        """
        Load and prepare qualification data for predictions.

        Args:
            data_path (str, optional): Path to folder with .csv files containing race data.
                                     Defaults to 'data' directory.

        Returns:
            pd.DataFrame: Prepared feature dataset containing driver performance metrics,
                         weather conditions, and team information.

        Raises:
            FileNotFoundError: If required files (lap_*.csv, weather_*.csv, result_*.csv) are not found
            ValueError: If multiple files match the same pattern or if data preparation fails
        """
        if data_path is None:
            data_path = 'data'
        try:
            # Define expected file patterns
            required_patterns = {
                'lap': os.path.join(data_path, '*lap*.csv'),
                'weather': os.path.join(data_path, '*weather*.csv'),
                'result': os.path.join(data_path, '*result*.csv')
            }

            # Collect all matching files
            found_files = {
                pattern: glob.glob(file_pattern)
                for pattern, file_pattern in required_patterns.items()
            }

            # Validate file counts
            for pattern, files in found_files.items():
                if len(files) == 0:
                    raise FileNotFoundError(
                        f"No {pattern} data files found in '{data_path}' directory. "
                        f"Expected file pattern: {required_patterns[pattern]}"
                    )
                elif len(files) > 1:
                    raise ValueError(
                        f"Multiple {pattern} data files found: {files}\n"
                        f"Expected exactly one file matching pattern: {required_patterns[pattern]}"
                    )

            # Load raw data and convert time columns to seconds
            self.lap_data = pd.read_csv(found_files['lap'][0])
            self.weather_data = pd.read_csv(found_files['weather'][0])
            self.result_data = pd.read_csv(found_files['result'][0])
            self.lap_data = self.convert_to_seconds(self.lap_data)

            try:
                # Filter out pit stop laps and select relevant performance metrics
                self._processed_lap_data = self.lap_data[
                    self.lap_data['PitOutTime'].isna() &
                    self.lap_data['PitInTime'].isna()
                    ][[
                    'Driver', 'LapTime', 'Sector1Time', 'Sector2Time',
                    'Sector3Time', 'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST',
                    'IsPersonalBest', 'Compound', 'TyreLife', 'FreshTyre', 'Team',
                    'TrackStatus', 'Deleted'
                ]]

                # Calculate average weather conditions across all sessions
                self._processed_weather = self.weather_data[
                    ['AirTemp', 'Humidity', 'Pressure', 'TrackTemp']
                ].mean()

                # Set Rainfall flag if any session had rain
                self._processed_weather['Rainfall'] = self.weather_data['Rainfall'].any()

                # Extract best performance metrics per driver
                best_times = self._processed_lap_data.groupby(['Driver'])[
                    ['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time']
                ].min().reset_index()

                max_speeds = self._processed_lap_data.groupby(['Driver'])[
                    ['SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST']
                ].max().reset_index()

                # Combine all features into final dataset
                self.feature_data = pd.merge(best_times, max_speeds, on=['Driver'])
                self.feature_data = pd.merge(
                    self.feature_data,
                    self.result_data[['Abbreviation', 'Position', 'TeamName']],
                    left_on=['Driver'],
                    right_on=['Abbreviation']
                )
                self.feature_data.drop(columns=['Abbreviation'], inplace=True)

                # Add weather conditions to all rows
                self.feature_data = pd.concat([
                    self.feature_data,
                    pd.DataFrame([self._processed_weather] * len(self.feature_data)).reset_index(drop=True)
                ], axis=1)

                return self.feature_data

            except Exception as e:
                raise ValueError(f"Error during data preparation: {str(e)}") from e

        except Exception as e:
            raise type(e)(
                f"Error loading data from {data_path}: {str(e)}\n"
                "Please ensure all required files exist in the specified directory."
            ) from e

    def predict_n_stint(self, X):
        """
        Predict the number of stints for each driver.

        Args:
            X (pd.DataFrame): DataFrame containing driver and team information,
                            performance metrics, and weather conditions.

        Returns:
            np.array: Predicted number of stints for each driver (integers)
        """
        # Create a copy to avoid modifying the input
        X_prep = X.copy()

        # Encode categorical variables
        for col in ['Driver', 'TeamName']:
            le = self.scalers['label_encoders_n_stint'][col]
            X_prep[col] = le.transform(X_prep[col])

        # Ensure Position is integer
        X_prep['Position'] = X_prep['Position'].astype(int)

        # Predict and round to nearest integer
        n_stint_pred = np.round(self.models['n_stint'].predict(X_prep)).astype(int)

        return n_stint_pred

    def predict_compound(self, X, n_stint_pred):
        """
        Predict compound choices for all stints of each driver.

        Args:
            X (pd.DataFrame): DataFrame containing driver and team information
            n_stint_pred (np.array): Predicted number of stints for each driver

        Returns:
            np.array: Predicted compound choices for each driver's stints
        """
        # Create a copy to avoid modifying the input
        X_prep = X.copy()

        X_prep = X_prep.drop(columns=['LapTime', 'Sector1Time', 'Sector2Time',
                                      'Sector3Time', 'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST'])

        # Create rows for each driver's stints
        driver_stints = []
        for driver, n_stints in zip(X['Driver'], n_stint_pred):
            for stint in range(1, n_stints + 1):
                driver_stints.append([driver, stint])
        driver_stints = pd.DataFrame(driver_stints, columns=['Driver', 'Stint'])

        # Add stint information to features
        X_prep = pd.merge(X_prep, driver_stints, on='Driver', how='outer')

        # Encode categorical variables
        for col in ['Driver', 'TeamName']:
            le = self.scalers['label_encoders_comp'][col]
            X_prep[col] = le.transform(X_prep[col])

        # Predict compounds and decode back to names
        compound_encoded = self.models['compound'].predict(X_prep)
        compound_pred = self.scalers['label_encoders_comp']['Compound'].inverse_transform(compound_encoded)

        return compound_pred

    def predict_pit_lap(self, X, n_stint_pred):
        """
        Predict pit stop laps for each stint of each driver.

        Args:
            X (pd.DataFrame): DataFrame containing driver and team information
            n_stint_pred (np.array): Predicted number of stints for each driver

        Returns:
            np.array: Predicted pit stop lap numbers for each driver's stints
        """
        # Create a copy to avoid modifying the input
        X_prep = X.copy()

        X_prep = X_prep.drop(columns=['LapTime', 'Sector1Time', 'Sector2Time',
                                      'Sector3Time', 'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST'])

        # Create rows for each driver's stints
        driver_stints = []
        for driver, n_stints in zip(X['Driver'], n_stint_pred):
            for stint in range(1, n_stints + 1):
                driver_stints.append([driver, stint])
        driver_stints = pd.DataFrame(driver_stints, columns=['Driver', 'Stint'])

        # Add stint information to features
        X_prep = pd.merge(X_prep, driver_stints, on='Driver', how='outer')

        # Encode categorical variables
        for col in ['Driver', 'TeamName']:
            le = self.scalers['label_encoders_comp'][col]
            X_prep[col] = le.transform(X_prep[col])

        # Predict pit stop laps
        pit_lap_pred = np.round(self.models['pit_lap'].predict(X_prep)).astype(int)

        return pit_lap_pred

    def predict_avg_lap_time(self, X, n_stint_pred, compound_pred, pit_lap_pred):
        """
        Predict average lap times for each stint of each driver.

        Args:
            X (pd.DataFrame): DataFrame containing driver and team information
            n_stint_pred (np.array): Predicted number of stints
            compound_pred (np.array): Predicted compound choices
            pit_lap_pred (np.array): Predicted pit stop laps

        Returns:
            np.array: Predicted average lap times for each stint
        """
        # Create a copy to avoid modifying the input
        X_prep = X.copy()

        # Create rows for each driver's stints
        driver_stints = []
        for driver, n_stints in zip(X['Driver'], n_stint_pred):
            for stint in range(1, n_stints + 1):
                driver_stints.append([driver, stint])
        driver_stints = pd.DataFrame(driver_stints, columns=['Driver', 'Stint'])

        # Add stint information and previous predictions
        X_prep = pd.merge(X_prep, driver_stints, on='Driver', how='outer')
        X_prep['Compound'] = compound_pred
        X_prep['Pit lap'] = pit_lap_pred

        # Calculate tyre life for each stint
        X_prep['Tyre life'] = np.where(
            X_prep['Stint'] == 1,
            X_prep['Pit lap'],
            X_prep['Pit lap'] - X_prep['Pit lap'].shift(1))

        # Encode categorical variables
        for col in ['Driver', 'TeamName', 'Compound']:
            le = self.scalers['label_encoders_lap'][col]
            X_prep[col] = le.transform(X_prep[col])

        # Predict average lap times
        avg_lap_time_pred = self.models['lap_time'].predict(X_prep)

        return avg_lap_time_pred

    def run_pipeline(self, X, df_as_output: bool = True):
        """
        Run the full race strategy prediction pipeline on input data.

        This method orchestrates the prediction of multiple race strategy components:
        1. Number of pit stops/stints for each driver
        2. Tire compound selection for each stint
        3. Pit stop lap predictions
        4. Average lap time predictions

        Args:
            X (pd.DataFrame): Input DataFrame containing race and driver features.
                Must include a 'Driver' column.

            df_as_output (bool, optional): Format of return value. Defaults to True.
                - If True: Returns a single DataFrame with all predictions.
                - If False: Returns individual prediction arrays.

        Returns:
            if df_as_output=True:
                pd.DataFrame: A DataFrame containing all predictions with columns:
                    - Driver: Driver name/identifier
                    - Stint: Stint number (1 to n_stints)
                    - N Stints: Total number of stints predicted
                    - Compound: Predicted tire compound for the stint
                    - Pit lap: Predicted lap number for pit stop
                    - Avg lap time: Predicted average lap time
                    - Tyre life: Number of laps on the tire compound

            if df_as_output=False:
                tuple: A tuple containing:
                    - stints_list (list): Predicted number of stints per driver
                    - compounds_list (list): Predicted tire compounds
                    - pits_list (list): Predicted pit stop laps
                    - avg_lap_time (list): Predicted average lap times

        Examples:
            >>> # Get predictions as a DataFrame
            >>> predictions_df = model.run_pipeline(test_data)
            >>>
            >>> # Get raw prediction arrays
            >>> stints, compounds, pits, lap_times = model.run_pipeline(
            ...     test_data, df_as_output=False
            ... )

        Notes:
            - All component predictions (stints, compounds, etc.) are made sequentially,
                with each step potentially using predictions from previous steps
            - The tyre life is calculated as the difference between consecutive pit stops
                (or first pit stop for stint 1)
        """
        stints_list = self.predict_n_stint(X)
        compounds_list = self.predict_compound(X, n_stint_pred=stints_list)
        pits_list = self.predict_pit_lap(X, n_stint_pred=stints_list)
        avg_lap_time = self.predict_avg_lap_time(X, n_stint_pred=stints_list, compound_pred=compounds_list,
                                                 pit_lap_pred=pits_list)
        if df_as_output:
            driver_stints = []
            for driver, n_stints in zip(X['Driver'], stints_list):
                for stint in range(1, n_stints + 1):
                    driver_stints.append([driver, stint, n_stints])
            res_df = pd.DataFrame(driver_stints, columns=['Driver', 'Stint', 'N Stints'])
            comp_pit_time_df = pd.DataFrame([compounds_list, pits_list, avg_lap_time],
                                            index=['Compound', 'Pit lap', 'Avg lap time']).T
            res_df = pd.concat([res_df, comp_pit_time_df], axis=1)
            res_df['Tyre life'] = np.where(
                res_df['Stint'] == 1,
                res_df['Pit lap'],
                res_df['Pit lap'] - res_df['Pit lap'].shift(1))
            return res_df
        else:
            return stints_list, compounds_list, pits_list, avg_lap_time
