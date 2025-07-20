from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder.appName("PowerPredictionModel").getOrCreate()


data = spark.read.parquet(r"D:\Programming\Data_Engineering\Apache_Spark\project\power_consumption_spark\Data\featured_data")



feature_cols = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5',
                'feature_6', 'feature_7', 'feature_8', 'feature_9', 'feature_10',
                'feature_11', 'feature_12', 'feature_13', 'feature_14', 'hour', 
                'HOLIDAY', 'month', 'day_of_week', 
                'is_weekend', 'day_period_index','ENERGY']
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
data = assembler.transform(data)


train_df, test_df = data.randomSplit([0.8, 0.2], seed=42)

train_data = train_df.select("features", "ENERGY")
test_data = test_df.select("features", "ENERGY")

print("Train count:", train_df.count())
print("Test count:", test_df.count())

lr = LinearRegression(featuresCol="features", labelCol="ENERGY")
lr_model = lr.fit(train_data)

test_predictions = lr_model.transform(test_data)

evaluator = RegressionEvaluator(labelCol="ENERGY", predictionCol="prediction", metricName="rmse")
rmse_test = evaluator.evaluate(test_predictions)
print(f"Test RMSE = {rmse_test}")
