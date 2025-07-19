from pyspark.sql import SparkSession
from pyspark.sql.functions import col, dayofweek, month, to_date, when
from pyspark.ml.feature import StringIndexer
import random

spark = SparkSession.builder.getOrCreate()

df = spark.read.csv("Data/processed_csv", header=True, inferSchema=True)

df = df.withColumn("DATE", to_date(col("DATE"), "yyyy-MM-dd"))

df = df.withColumn("month", month(col("DATE")))
df = df.withColumn("day_of_week", dayofweek(col("DATE")))  

df = df.withColumn("is_weekend", when(col("day_of_week").isin([1, 7]), 1).otherwise(0))

df = df.withColumn("HOLIDAY", when(col("HOLIDAY") == True, 1).otherwise(0))

day_period_indexer = StringIndexer(inputCol="day_period", outputCol="day_period_index")
df = day_period_indexer.fit(df).transform(df)


timestamp_col = "DATE"

other_columns = [col for col in df.columns if col != timestamp_col]
selected_columns = random.sample(other_columns, 9)

selected_columns = [timestamp_col] + selected_columns

sampled_rows = df.select(selected_columns).sample(withReplacement=False, fraction=0.1, seed=42).limit(50)

print(sampled_rows.show(30))

df.write.mode("overwrite").parquet("Data/featured_data")

