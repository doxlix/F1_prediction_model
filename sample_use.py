from download_data import DataDownloader
from model_class import F1PredictionPipeline


def main():
    # Create an instance
    data_get = DataDownloader()

    data_get.download_data(year=2023, location='Mexico', event='qualifying')
    data_get.save_data()

    data_get.download_data(year=2023, location='Mexico', event='race')
    data_get.save_data()

    predictor = F1PredictionPipeline()
    data = predictor.load_data()

    prediction_df = predictor.run_pipeline(data)
    prediction_df.to_csv('PREDICTION.csv', index=False)

    predictions = predictor.run_pipeline(data, df_as_output=False)
    for name, i in zip(('Pit amount', 'Compound', 'Pit lap number', 'Average time in sec'), predictions):
        print(f"{name}:\n{i}")

    # F1PredictionPipeline.convert_to_seconds() <- if you need to convert from datetime to seconds


if __name__ == "__main__":
    main()