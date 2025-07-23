from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
import random
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType
spark = SparkSession.builder.appName("GBTModelPrediction").getOrCreate()

model_path = "D:\Programming\Data_Engineering\Apache_Spark\project\power_consumption_spark\Models\Model_Spark_v1"

model = PipelineModel.load(model_path)

for i, stage in enumerate(model.stages):
    print(f"Stage {i}: {stage.__class__.__name__}")

numeric_cols = [
    'HDD18_3', 'CDD0', 'CDD10', 'PRECTOT', 'RH2M',
    'T2M', 'T2M_MIN', 'T2M_MAX', 'ALLSKY',
    'day_of_week', 'month', 'year',
    'temp_diff', 'temp_avg', 'HOLIDAY',
    'ENERGY_lag1', 'ENERGY_lag2' ,
    'energy_ma_3', 'energy_std_3'
]

data = spark.read.parquet(r"Data\featured_data")

df_sample = data.sample(withReplacement=False, fraction=0.05, seed=42)


def add_noise(value, intensity=0.10):
    if value is None:
        return None
    noise = random.gauss(0, abs(value) * intensity)
    return round(value + noise, 2)

add_noise_udf = udf(lambda x: add_noise(x, intensity=0.05), DoubleType())

df_sample = df_sample.select(numeric_cols)

for col_name in numeric_cols:
    df_sample = df_sample.withColumn(col_name, add_noise_udf(col(col_name)))

print(df_sample.show(10))


predictions = model.transform(df_sample)

print(predictions.select("prediction").show(truncate=False))



