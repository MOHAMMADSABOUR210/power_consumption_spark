from pyspark.sql import SparkSession
from pyspark.sql.functions import  hour, dayofweek, month
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler 
import numpy as np

spark = SparkSession.builder.appName("ElectricityPreprocessing").getOrCreate()

df = spark.read.csv(r"D:\Programming\Data_Engineering\Apache_Spark\project\power_consumption_spark\Data\power.csv", header=True, inferSchema=True)

print("Schema:")
for field in df.schema.fields:
    print(f"{field.name}: {field.dataType}")


print(df.show(5))


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


def cosine_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def remove_similar_rows(df, threshold=0.98, sample_fraction=0.01):
    from pyspark.sql.types import NumericType

    numeric_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, NumericType)]
    sample_df = df.select(numeric_cols).sample(withReplacement=False, fraction=sample_fraction, seed=42)
    original_rows = df.sample(withReplacement=False, fraction=sample_fraction, seed=42).collect()
    numeric_rows = sample_df.collect()

    seen = []
    result = []

    for i, row in enumerate(numeric_rows):
        values = np.array(row)
        is_duplicate = False
        for seen_row in seen:
            if cosine_similarity(values, seen_row) > threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            seen.append(values)
            result.append(original_rows[i])  

    cleaned_df = spark.createDataFrame(result, df.schema)
    return cleaned_df


print(f"Total number of rows Before removing similar rows: {df.count()}")
df = remove_similar_rows(df, threshold=0.98, sample_fraction=0.01)
print(f"Total number of rows After removing similar rows: {df.count()}")
