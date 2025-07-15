from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp, hour, dayofweek, month

spark = SparkSession.builder.appName("ElectricityPreprocessing").getOrCreate()

df = spark.read.csv(r"D:\Programming\Data_Engineering\Apache_Spark\project\power_consumption_spark\Data\power.csv", header=True, inferSchema=True)

print(df.printSchema())
print(df.show(5))


df = df.withColumn("datetime", to_timestamp("datetime", "yyyy-MM-dd HH:mm:ss"))

df = df.withColumn("hour", hour("datetime"))
df = df.withColumn("day_of_week", dayofweek("datetime"))
df = df.withColumn("month", month("datetime"))


df = df.dropna()

print(f"Total rows after cleaning: {df.count()}")
print(df.show(5))


df.write.mode("overwrite").parquet(r"D:\Programming\Data_Engineering\Apache_Spark\project\power_consumption_spark\Data\processed.parquet")

spark.stop()
