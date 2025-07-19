from pyspark.sql import SparkSession
from pyspark.sql.functions import  hour, dayofweek, month,stddev,mean , when,to_timestamp
from pyspark.ml.feature import MinMaxScaler,VectorAssembler 
import random
from pyspark.sql.functions import col as col_funs
from pyspark.ml.functions import vector_to_array
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

df = df.withColumn("hour", hour("DATE"))
df = df.withColumn("day_of_week", dayofweek("DATE"))
df = df.withColumn("month", month("DATE"))


print("Befor dropna : ",df.count())
df = df.dropna()
print("after dropna : ",df.count())


print("Before dropping duplicates: ", df.count())
df = df.dropDuplicates()
print("After dropping duplicates: ", df.count())


time_col = "DATE"

cols = df.columns
numeric_cols = [c for c in cols if c != time_col]


print(f"Befor remove outlier rows : {df.count()}")

for col in numeric_cols:
    stats = df.select(mean(col), stddev(col)).first()
    mean_val, std_val = stats
    if std_val is not None and std_val != 0:
        df = df.filter((df[col] >= mean_val - 3 * std_val) & (df[col] <= mean_val + 3 * std_val))

print(f"After remove outlier rows : {df.count()}")


assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features_vec")
df = assembler.transform(df)

scaler = MinMaxScaler(inputCol="features_vec", outputCol="scaled_features")
df = scaler.fit(df).transform(df)


df = df.withColumn("features_array", vector_to_array(col_funs("scaled_features")))

num_features = len(numeric_cols)
for i in range(num_features):
    df = df.withColumn(f"feature_{i+1}", col_funs("features_array")[i])

print(df.select([f"feature_{i+1}" for i in range(num_features)]).show(5))



print(df.select("hour").show(10))  
df = df.withColumn("day_period", 
                   when(df["hour"] < 6, "night")
                   .when(df["hour"] < 12, "morning")
                   .when(df["hour"] < 18, "afternoon")
                   .otherwise("evening"))
print(df.select("day_period").show(10))


final_cols = [f"feature_{i+1}" for i in range(num_features)] + [
    "hour", "day_period", "DATE", "ENERGY","HOLIDAY"
]

output_path = "Data/processed_csv"
if os.path.exists(output_path):
    shutil.rmtree(output_path)
df.select(final_cols).write.option("header", True).mode("overwrite").csv(output_path)