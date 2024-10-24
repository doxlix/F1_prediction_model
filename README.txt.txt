model_class - contains pipeline to:
1 load models from /models folder
2 encoders from /encoders folder
3 data from /data folder
4 make predictions on qualification data
5 return df or arrays with predicted features

download_data - contains class to download data using fastf1 API
You can put your own files into data folder, but there's has to be:
 - lap data
 - weather data
 - results data
Only files which has "lap", "weather" or "results" in the name will be read

Only one file with "lap" in it's name is allowed
Only one file with "weather" in it's name is allowed
Only one file with "results" in it's name is allowed

If you need any additional data put it into separate folder or name it differently

download_data will automatically put qualifications data in /data folder and races data in /data/race folder

Rating functions are not present



