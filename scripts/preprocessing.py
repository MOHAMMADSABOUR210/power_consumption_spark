from pyspark.sql import SparkSession
from pyspark.sql.functions import  stddev,mean ,to_timestamp
import random
import shutil
import os


spark = SparkSession.builder.appName("ElectricityPreprocessing").getOrCreate()



df1 = spark.read.csv(r"D:\Programming\Data_Engineering\Apache_Spark\project\power_consumption_spark\Data\db_building_A.csv", header=True, inferSchema=True)
df2 = spark.read.csv(r"D:\Programming\Data_Engineering\Apache_Spark\project\power_consumption_spark\Data\db_building_B.csv", header=True, inferSchema=True)

df = df1.unionByName(df2)

print("Schema:")
for field in df.schema.fields:
    print(f"{field.name}: {field.dataType}")


print("50 sample with 10 columns: ")
timestamp_col = "DATE"

other_columns = [col for col in df.columns if col != timestamp_col]
selected_columns = random.sample(other_columns, 11)

selected_columns = [timestamp_col] + selected_columns

sampled_rows = df.select(selected_columns).sample(withReplacement=False, fraction=0.1, seed=42).limit(50)

print(df.show(50))

df = df.withColumn("DATE", to_timestamp("DATE", "M/d/yyyy H:mm"))

print("Befor dropna : ",df.count())
df = df.dropna()
print("after dropna : ",df.count())


print("Before dropping duplicates: ", df.count())
df = df.dropDuplicates()
print("After dropping duplicates: ", df.count())


time_col = ["DATE","HOLIDAY"]

cols = df.columns
numeric_cols = [c for c in cols if c not in time_col]


print(f"Befor remove outlier rows : {df.count()}")

for col in numeric_cols:
    stats = df.select(mean(col), stddev(col)).first()
    mean_val, std_val = stats
    if std_val is not None and std_val != 0:
        df = df.filter((df[col] >= mean_val - 3 * std_val) & (df[col] <= mean_val + 3 * std_val))

print(f"After remove outlier rows : {df.count()}")


final_cols = df.columns
print(final_cols)

output_path = "Data/processed"
if os.path.exists(output_path):
    shutil.rmtree(output_path)
df.write.mode("overwrite").parquet(output_path)
