# Project Overview

This project contains tools to load models, process data, and make predictions on Formula 1 qualification data. Below is an overview of the main classes and their functionality.

## model_class

This class contains a pipeline to:
1. Load models from the `/models` folder.
2. Load encoders from the `/encoders` folder.
3. Load data from the `/data` folder.
4. Make predictions on qualification data.
5. Return a DataFrame or arrays with predicted features.

## download_data

This class is responsible for downloading data using the FastF1 API. 

### Data Requirements:
You can put your own files into the `/data` folder, but the following files are required:
- **Lap Data**: Files containing "lap" in the name.
- **Weather Data**: Files containing "weather" in the name.
- **Results Data**: Files containing "results" in the name.

### Important Rules:
- Only one file with "lap" in its name is allowed.
- Only one file with "weather" in its name is allowed.
- Only one file with "results" in its name is allowed.

If you need any additional data, place it into a separate folder or ensure it is named differently to avoid conflicts.

`download_data` will automatically store:
- Qualification data in the `/data` folder.
- Race data in the `/data/race` folder.

### Note:
- Rating functions are not included in this project.
