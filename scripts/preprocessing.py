from pyspark.sql import SparkSession
from pyspark.sql.functions import  hour, dayofweek, month,stddev,mean ,udf
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import MinMaxScaler,VectorAssembler 
import random
from pyspark.ml.linalg import Vectors, DenseVector
from pyspark.sql.types import FloatType



spark = SparkSession.builder.appName("ElectricityPreprocessing").getOrCreate()

df = spark.read.csv(r"D:\Programming\Data_Engineering\Apache_Spark\project\power_consumption_spark\Data\power.csv", header=True, inferSchema=True)

print("Schema:")
for field in df.schema.fields:
    print(f"{field.name}: {field.dataType}")


print("100 sample with 10 columns: ")
timestamp_col = "datetime"

other_columns = [col for col in df.columns if col != timestamp_col]
selected_columns = random.sample(other_columns, 10)

selected_columns = [timestamp_col] + selected_columns

sampled_rows = df.select(selected_columns).sample(withReplacement=False, fraction=0.1, seed=42).limit(100)

print(sampled_rows.show(100))

df = df.withColumn("hour", hour("datetime"))
df = df.withColumn("day_of_week", dayofweek("datetime"))
df = df.withColumn("month", month("datetime"))

print("Befor dropna : ",df.count())
df = df.dropna()
print("after dropna : ",df.count())


def drop_highly_correlated_columns(df, threshold=0.95):

    cols = [col for col, dtype in df.dtypes if dtype in ('double', 'int', 'float', 'long')]
    
    assembler = VectorAssembler(inputCols=cols, outputCol="features")
    vector_df = assembler.transform(df).select("features")

    corr_matrix = Correlation.corr(vector_df, "features").head()[0].toArray()

    to_drop = set()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if abs(corr_matrix[i][j]) >= threshold and cols[j] not in to_drop:
                to_drop.add(cols[j])

    return df.drop(*to_drop)


print("Before dropping highly correlated columns: ", len(df.columns))

df = drop_highly_correlated_columns(df, threshold=0.9)

print("Columns after dropping highly correlated ones: ", len(df.columns))


print("Before dropping duplicates: ", df.count())
df = df.dropDuplicates()
print("After dropping duplicates: ", df.count())


time_col = "datetime"

cols = df.columns
numeric_cols = [c for c in cols if c != time_col]


print("Before dropping constant columns: ", len(df.columns))

for col in numeric_cols:
    
    std = df.select(stddev(col)).first()[0]
    if std == 0 or std is None:
        df = df.drop(col)

print("After dropping constant columns: ", len(df.columns))
print(df.columns)



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


min_udf = udf(lambda vec: float(min(vec)), FloatType())
max_udf = udf(lambda vec: float(max(vec)), FloatType())

df.select(min_udf("scaled_features").alias("min_val"), max_udf("scaled_features").alias("max_val")).show()