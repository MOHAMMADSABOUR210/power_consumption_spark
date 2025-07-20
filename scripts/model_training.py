from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import datetime

spark = SparkSession.builder.appName("PowerPredictionModel").getOrCreate()


data = spark.read.parquet(r"Data\featured_data")


feature_cols = ['scaled_features','ENERGY']
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


evaluator_mae = RegressionEvaluator(labelCol="ENERGY", predictionCol="prediction", metricName="mae")
mae = evaluator_mae.evaluate(test_predictions)
print(f"Test MAE = {mae}")


evaluator_r2 = RegressionEvaluator(labelCol="ENERGY", predictionCol="prediction", metricName="r2")
r2 = evaluator_r2.evaluate(test_predictions)
print(f"Test R2 = {r2}")


now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = fr"D:\Programming\Data_Engineering\Apache_Spark\project\power_consumption_spark\Model_Spark_{now}"
lr_model.save(model_path)