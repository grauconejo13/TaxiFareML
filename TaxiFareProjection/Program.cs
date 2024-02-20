using System;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

public class TaxiFare
{
    [LoadColumn(0)] public string? Vendor_id;
    [LoadColumn(1)] public float Rate_code;
    [LoadColumn(2)] public float Passenger_count;
    [LoadColumn(3)] public float Trip_time_in_secs;
    [LoadColumn(4)] public float Trip_distance;
    [LoadColumn(5)] public string? Payment_type;
    [LoadColumn(6)] public float Fare_amount;
}

public class TaxiFareProjection
{
    [ColumnName("Score")] public float Fare_amount;
}

class Program
{
    static void Main(string[] args)
    {
        var context = new MLContext(seed: 0);

        var dataPath = Path.Combine(AppContext.BaseDirectory, "taxi-fare-train.csv");

        var data = context.Data.LoadFromTextFile<TaxiFare>(dataPath, separatorChar: ',');

        var trainTestSplit = context.Data.TrainTestSplit(data, testFraction: 0.2);

        var pipeline = context.Transforms.CopyColumns("Label", nameof(TaxiFare.Fare_amount))
        .Append(context.Transforms.Categorical.OneHotEncoding(nameof(TaxiFare.Vendor_id)))
        .Append(context.Transforms.Categorical.OneHotEncoding(nameof(TaxiFare.Payment_type)))
        .Append(context.Transforms.Concatenate("Features",
            nameof(TaxiFare.Rate_code),
            nameof(TaxiFare.Passenger_count),
            nameof(TaxiFare.Trip_time_in_secs),
            nameof(TaxiFare.Trip_distance),
            nameof(TaxiFare.Vendor_id),
            nameof(TaxiFare.Payment_type)))
        .Append(context.Regression.Trainers.Sdca(new SdcaRegressionTrainer.Options
        {
            MaximumNumberOfIterations = 25,
            ConvergenceTolerance = 0.001f,
            L2Regularization = 0.1f,
            LabelColumnName = "Label"
        }));

        var model = pipeline.Fit(trainTestSplit.TrainSet);

        var testMetrics = context.Regression.Evaluate(model.Transform(trainTestSplit.TestSet));

        Console.WriteLine($"RMS error: {testMetrics.RootMeanSquaredError}");
        Console.WriteLine($"RSquared: {testMetrics.RSquared}\n");

        var predictionEngine = context.Model.CreatePredictionEngine<TaxiFare, TaxiFareProjection>(model);

        // User inputs for prediction
        var inputs = new List<TaxiFare>
        {
            new TaxiFare
            {
                Vendor_id = "VTS",
                Rate_code = 1,
                Passenger_count = 6,
                Trip_time_in_secs = 900,
                Trip_distance = 5,
                Payment_type = "CRD"
            },
            new TaxiFare
            {
                Vendor_id = "CMT",
                Rate_code = 2,
                Passenger_count = 2,
                Trip_time_in_secs = 1200,
                Trip_distance = 7,
                Payment_type = "CRD"
            },
            new TaxiFare
            {
                Vendor_id = "CMT",
                Rate_code = 1,
                Passenger_count = 3,
                Trip_time_in_secs = 297,
                Trip_distance = 15,
                Payment_type = "CSH"
            }

        };


       
        foreach (var input in inputs)
        {
            // Measure the prediction time
             var watch = System.Diagnostics.Stopwatch.StartNew();
            // Make a prediction
            var prediction = predictionEngine.Predict(input);
            watch.Stop();
            var elapsedMs = watch.ElapsedMilliseconds;

            Console.WriteLine($"The predicted fare amount: ${prediction.Fare_amount.ToString("F2")}");
            Console.WriteLine($"Prediction took {elapsedMs} ms\n");
        }
    }
}
