from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("ElectricityPreprocessing").getOrCreate()

df = spark.read.csv("D:\Programming\Data_Engineering\Apache_Spark\project\power_consumption_spark\Data\power.csv", header=True, inferSchema=True)

print(df.printSchema())
print(df.show(5))
