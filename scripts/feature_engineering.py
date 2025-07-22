from pyspark.sql import SparkSession
from pyspark.sql.functions import col, dayofweek, month, to_date, when,hour,year,lag,avg, stddev
from pyspark.ml.feature import StringIndexer
import random
from pyspark.sql.window import Window

spark = SparkSession.builder.getOrCreate()

df = spark.read.parquet("Data/processed", header=True, inferSchema=True)


df = df.withColumn("hour", hour("DATE"))
df = df.withColumn("day_of_week", dayofweek("DATE"))
df = df.withColumn("month", month("DATE"))
df = df.withColumn("year", year("DATE"))


df = df.withColumn("DATE", to_date(col("DATE"), "yyyy-MM-dd"))

df = df.withColumn("is_weekend", when(col("day_of_week").isin([1, 7]), 1).otherwise(0))

df = df.withColumn("day_period", 
                   when(df["hour"] < 6, "night")
                   .when(df["hour"] < 12, "morning")
                   .when(df["hour"] < 18, "afternoon")
                   .otherwise("evening"))
print(df.select("day_period").show(10))

day_period_indexer = StringIndexer(inputCol="day_period", outputCol="day_period_index")
df = day_period_indexer.fit(df).transform(df)
df = df.drop("day_period")

df = df.withColumn(
    "season",
    when((col("month") >= 3) & (col("month") <= 5), "Spring")
    .when((col("month") >= 6) & (col("month") <= 8), "Summer")
    .when((col("month") >= 9) & (col("month") <= 11), "fall")
    .otherwise("Winter")
)

timestamp_col = "DATE"

other_columns = [col for col in df.columns if col != timestamp_col]
selected_columns = random.sample(other_columns, 9)

selected_columns = [timestamp_col] + selected_columns

sampled_rows = df.select(selected_columns).sample(withReplacement=False, fraction=0.1, seed=42).limit(50)

print(df.show(30))

df = df.withColumn("temp_diff", col("T2M_MAX") - col("T2M_MIN"))
df = df.withColumn("temp_avg", (col("T2M_MAX") + col("T2M_MIN")) / 2)


windowSpec = Window.orderBy("DATE")

df = df.withColumn("ENERGY_lag1", lag("ENERGY", 1).over(windowSpec))

df = df.withColumn("ENERGY_lag2", lag("ENERGY", 2).over(windowSpec))


windowSpec = Window.orderBy("DATE").rowsBetween(-2, 0)

df = df.withColumn("energy_ma_3", avg("ENERGY").over(windowSpec))
df = df.withColumn("energy_std_3", stddev("ENERGY").over(windowSpec))

df = df.na.drop(subset=["ENERGY_lag1", "ENERGY_lag2","energy_ma_3","energy_std_3"])

df.write.mode("overwrite").parquet("Data/featured_data")


print("Schema:")
for field in df.schema.fields:
    print(f"{field.name}: {field.dataType}")
