from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressionModel

# ساخت SparkSession
spark = SparkSession.builder.appName("GBTModelPrediction").getOrCreate()

model_path = "D:\Programming\Data_Engineering\Apache_Spark\project\power_consumption_spark\Models\Model_Spark_v1"

model = GBTRegressionModel.load(model_path)

print(model.featuresCol)

numeric_cols = [
    'HDD18_3', 'CDD0', 'CDD10', 'PRECTOT', 'RH2M',
    'T2M', 'T2M_MIN', 'T2M_MAX', 'ALLSKY',
    'day_of_week', 'month', 'year',
    'temp_diff', 'temp_avg', 'HOLIDAY',
    'ENERGY_lag1', 'ENERGY_lag2' ,
    'energy_ma_3', 'energy_std_3'
]

df_new = spark.createDataFrame(new_data, numeric_cols)

assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features")
df_new_vec = assembler.transform(df_new)

predictions = model.transform(df_new_vec)


predictions.select(*numeric_cols, "prediction").show()
