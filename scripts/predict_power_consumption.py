from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
import random
from pyspark.sql.functions import col, udf, lit
from pyspark.sql.types import DoubleType
spark = SparkSession.builder.appName("GBTModelPrediction").getOrCreate()

model_path = "D:\Programming\Data_Engineering\Apache_Spark\project\power_consumption_spark\Models\Model_Spark_v1"

model = PipelineModel.load(model_path)

for i, stage in enumerate(model.stages):
    print(f"Stage {i}: {stage.__class__.__name__}")

numeric_cols = [
    'energy_ma_3', 'ENERGY_lag1', 'ENERGY_lag2',
    'energy_std_3', 'HOLIDAY', 'HDD18_3','temp_avg'
]

data = spark.read.parquet(r"Data\featured_data")


def data_sample(data, numeric_cols):
    df_sample = data.sample(withReplacement=False, fraction=0.05, seed=42)

    if "season" in numeric_cols:
        extra_seasons = ["spring", "summer", "fall", "winter"]
        extra_df = data.sparkSession.createDataFrame([(s,) for s in extra_seasons], ["season"])
        
        other_cols = [c for c in df_sample.columns if c != "season"]
        for c in other_cols:
            extra_df = extra_df.withColumn(c, lit(None))
        
        df_sample = df_sample.unionByName(extra_df.select(df_sample.columns))

    def add_noise(value, intensity=0.10):
        try:
            if value is None:
                return None
            val = float(value)
            noise = random.gauss(0, abs(val) * intensity)
            return round(val + noise, 2)
        except:
            return None

    add_noise_udf = udf(add_noise, DoubleType())

    numeric_cols_filtered = [f.name for f in data.schema.fields if f.name in numeric_cols and isinstance(f.dataType, DoubleType)]

    for col_name in numeric_cols_filtered:
        df_sample = df_sample.withColumn(col_name, add_noise_udf(col(col_name)))

    df_sample = df_sample.select(numeric_cols)
    print("-"*50)
    print(df_sample.show(10))
    return df_sample

##################################### model v1 #################################

df_sample = data_sample(data,numeric_cols)
predictions = model.transform(df_sample)

print(predictions.select("prediction").show(truncate=False))



##################################### model v2 #################################

model_path = "D:\Programming\Data_Engineering\Apache_Spark\project\power_consumption_spark\Models\Model_Spark_v2"

model = PipelineModel.load(model_path)
numeric_cols = numeric_cols + ["season"]
for i, stage in enumerate(model.stages):
    print(f"Stage {i}: {stage.__class__.__name__}")



df_sample = data_sample(data,numeric_cols)

print(df_sample.columns)

for stage in model.stages:
    if stage.__class__.__name__ == "StringIndexerModel":
        print(f"StringIndexer input col: {stage.getInputCol()}")
print("*"*50)
print(df_sample.show(12))
predictions = model.transform(df_sample)

print(predictions.select("prediction").show(truncate=False))

