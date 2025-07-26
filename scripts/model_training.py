from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler,StringIndexer,MinMaxScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import RandomForestRegressor , GBTRegressor
import datetime
from pyspark.sql.functions import  avg
from pyspark.ml import Pipeline
import os
import shutil


spark = SparkSession.builder.appName("PowerPredictionModel").getOrCreate()


data = spark.read.parquet(r"Data\featured_data")

print(data.columns)

numeric_cols = [
    'energy_ma_3', 'ENERGY_lag1', 'ENERGY_lag2',
    'energy_std_3', 'HOLIDAY', 'HDD18_3','temp_avg'
]

print(data.select(numeric_cols).describe().show())


assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features_vec")
scaler = MinMaxScaler(inputCol="features_vec", outputCol="scaled_features")
gbt = GBTRegressor(featuresCol="scaled_features", labelCol="ENERGY", maxIter=100, maxDepth=5)

pipeline = Pipeline(stages=[assembler, scaler, gbt])

train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

for col_name in numeric_cols:
    print(data.stat.corr(col_name, "ENERGY"))


model = pipeline.fit(train_data)
predictions = model.transform(test_data)

evaluator_rmse = RegressionEvaluator(labelCol="ENERGY", predictionCol="prediction", metricName="rmse")
evaluator_mae = RegressionEvaluator(labelCol="ENERGY", predictionCol="prediction", metricName="mae")
evaluator_r2 = RegressionEvaluator(labelCol="ENERGY", predictionCol="prediction", metricName="r2")

rmse = evaluator_rmse.evaluate(predictions)
mae = evaluator_mae.evaluate(predictions)
r2 = evaluator_r2.evaluate(predictions)

print(f"Test RMSE = {rmse}")
print(f"Test MAE = {mae}")
print(f"Test R2 = {r2}")

print(predictions.select("DATE", "ENERGY", "prediction").show(10, truncate=False))

model_path = fr"D:\Programming\Data_Engineering\Apache_Spark\project\power_consumption_spark\Models\Model_Spark_v1"
if os.path.exists(model_path):
    shutil.rmtree(model_path)
model.save(model_path)

#################################################prediction season ##############################################


season_indexer = StringIndexer(inputCol="season", outputCol="season_index")
final_assembler = VectorAssembler(inputCols=["scaled_features", "season_index"], outputCol="final_features")
gbt = GBTRegressor(featuresCol="final_features", labelCol="ENERGY", maxIter=100, maxDepth=5)

pipeline = Pipeline(stages=[assembler, scaler, season_indexer, final_assembler, gbt])

train_df, test_df = data.randomSplit([0.8, 0.2], seed=42)

model = pipeline.fit(train_df)

predictions = model.transform(test_df)

seasonal_energy = predictions.groupBy("season").agg(
    avg("ENERGY").alias("avg_energy_actual"),
    avg("prediction").alias("avg_energy_prediction")
)
print(seasonal_energy.show())

evaluator = RegressionEvaluator(
    labelCol="ENERGY", predictionCol="prediction", metricName="rmse"
)

rmse = evaluator.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE): {rmse}")

model_path = fr"D:\Programming\Data_Engineering\Apache_Spark\project\power_consumption_spark\Models\Model_Spark_v2"
if os.path.exists(model_path):
    shutil.rmtree(model_path)
model.save(model_path)


