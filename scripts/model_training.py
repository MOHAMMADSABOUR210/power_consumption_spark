from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler,StringIndexer,MinMaxScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import datetime
from pyspark.sql.functions import  avg
from pyspark.ml import Pipeline

spark = SparkSession.builder.appName("PowerPredictionModel").getOrCreate()


data = spark.read.parquet(r"Data\featured_data")


numeric_cols = ['HDD18_3', 'CDD0', 'CDD10', 'PRECTOT', 'RH2M', 'T2M', 'T2M_MIN', 'T2M_MAX', 'ALLSKY']

assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features_vec")
scaler = MinMaxScaler(inputCol="features_vec", outputCol="scaled_features")
final_assembler = VectorAssembler(inputCols=["scaled_features"], outputCol="final_features")
lr = LinearRegression(featuresCol="final_features", labelCol="ENERGY")

pipeline = Pipeline(stages=[assembler, scaler, final_assembler, lr])

train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)


print("Train count:", train_data.count())
print("Test count:", test_data.count())

lr = LinearRegression(featuresCol="features", labelCol="ENERGY")
lr_model = pipeline.fit(train_data)


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

print(test_predictions.select("ENERGY", "prediction").show(10))

now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = fr"D:\Programming\Data_Engineering\Apache_Spark\project\power_consumption_spark\Models\Model_Spark_{now}"
lr_model.save(model_path)

#################################################prediction season ##############################################


season_indexer = StringIndexer(inputCol="season", outputCol="season_index")
final_assembler = VectorAssembler(inputCols=["scaled_features", "season_index"], outputCol="final_features")
lr = LinearRegression(featuresCol="final_features", labelCol="ENERGY")

pipeline = Pipeline(stages=[assembler, scaler, season_indexer, final_assembler, lr])

train_df, test_df = data.randomSplit([0.8, 0.2], seed=42)

model = pipeline.fit(train_df)

predictions = model.transform(test_df)

seasonal_energy = predictions.groupBy("season").agg(avg("prediction").alias("avg_energy_prediction"))

print(seasonal_energy.show())

evaluator = RegressionEvaluator(
    labelCol="ENERGY", predictionCol="prediction", metricName="rmse"
)

rmse = evaluator.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE): {rmse}")

now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = fr"D:\Programming\Data_Engineering\Apache_Spark\project\power_consumption_spark\Models\Model_Spark_{now}"
model.save(model_path)

######################################Predicte New Rows###########################################

# from pyspark.sql import Row

# new_data = [
#     Row(HDD18_3=0.0, CDD0=10.0, CDD10=5.0, PRECTOT=1.2, RH2M=45.0,
#         T2M=30.0, T2M_MIN=24.0, T2M_MAX=36.0, ALLSKY=0.3, season="summer")
# ]
# new_df = spark.createDataFrame(new_data)

# prediction_new = model.transform(new_df)
# prediction_new.select("season", "prediction").show()