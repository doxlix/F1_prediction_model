import fastf1
import os
import pandas as pd


class DataDownloader:
    """
    A class to download and save Formula 1 session data using the FastF1 library.
    
    This class provides functionality to:
        - Download lap times, weather data, and race results
        - Save the downloaded data to CSV files
        - Handle different session types (qualifying, race, practice)
    
    Attributes:
        laps_data (Optional[pd.DataFrame]): DataFrame containing lap timing data
        weather_data (Optional[pd.DataFrame]): DataFrame containing weather conditions
        results_data (Optional[pd.DataFrame]): DataFrame containing session results
    
    Examples:
        >>> downloader = DataDownloader()
        >>> downloader.download_data(2023, 'Monaco', 'qualifying')
        >>> downloader.save_data(default_folder='monaco_data')
    """
    
    def __init__(self) -> None:
        """
        Initialize the DataDownloader with empty data containers.
        
        The containers are set to None initially and will be populated
        when download_data() is called.
        """
        self.laps_data: Optional[pd.DataFrame] = None
        self.weather_data: Optional[pd.DataFrame] = None
        self.results_data: Optional[pd.DataFrame] = None
    
    def download_data(
        self, 
        year: int, 
        location: str, 
        event: str = 'qualifying'
    ) -> None:
        """
        Download F1 session data for a specific race event.
        
        Args:
            year: The year of the race (e.g., 2023)
            location: The location/circuit of the race (e.g., 'Monaco', 'Silverstone')
            event: The session type. Defaults to 'qualifying'.
                  Valid options: 'qualifying', 'race', 'sprint', 'practice1', 'practice2', 'practice3'
        
        Raises:
            ValueError: If the year is invalid or location is not found
            ConnectionError: If unable to connect to the FastF1 API
            
        Examples:
            >>> downloader = DataDownloader()
            >>> # Download qualifying data from Monaco 2023
            >>> downloader.download_data(2023, 'Monaco', 'qualifying')
            >>> # Download race data from Silverstone 2023
            >>> downloader.download_data(2023, 'Silverstone', 'race')
        """
        session = fastf1.get_session(year, location, event)
        session.load(telemetry=False, messages=False)
        self._race_type = event
        self.laps_data = session.laps
        self.weather_data = session.weather_data
        self.results_data = session.results

    def save_data(self, default_folder: str = 'data') -> None:
        """
        Save the downloaded data to CSV files in the specified folder.
        
        The method creates three CSV files:
            - lap_data_{event}.csv: Contains lap timing information
            - weather_data_{event}.csv: Contains weather conditions
            - results_data_{event}.csv: Contains session results
        
        For non-qualifying sessions, a subfolder is created with the event name.
        
        Args:
            default_folder: Base directory path where files will be saved.
                          Defaults to 'data'.
        
        Raises:
            FileNotFoundError: If the specified path is invalid
            PermissionError: If writing to the specified path is not allowed
            AttributeError: If data hasn't been downloaded yet
            
        Examples:
            >>> downloader = DataDownloader()
            >>> downloader.download_data(2023, 'Monaco', 'qualifying')
            >>> # Save data to default folder
            >>> downloader.save_data()
            >>> # Save data to custom folder
            >>> downloader.save_data('monaco_qualifying_data')
            
        File Structure:
            For qualifying:
            data/
                lap_data_qualifying.csv
                weather_data_qualifying.csv
                results_data_qualifying.csv
                
            For other sessions (e.g., race):
            data/
                race/
                    lap_data_race.csv
                    weather_data_race.csv
                    results_data_race.csv
        """
        if self._race_type.lower() not in ['q', 'qualifying']:
            default_folder = os.path.join(default_folder, self._race_type)
        os.makedirs(default_folder, exist_ok=True) 
        self.laps_data.to_csv(os.path.join(default_folder, f'lap_data_{self._race_type}.csv'), index=False)
        self.weather_data.to_csv(os.path.join(default_folder, f'weather_data_{self._race_type}.csv'), index=False)
        self.results_data.to_csv(os.path.join(default_folder, f'results_data_{self._race_type}.csv'), index=False)
