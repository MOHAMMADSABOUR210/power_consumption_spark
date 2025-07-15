import os
os.environ["PYSPARK_PYTHON"] = r"D:\Programming\Data_Engineering\Apache_Spark\spark-env\Scripts\python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = r"D:\Programming\Data_Engineering\Apache_Spark\spark-env\Scripts\python.exe"

from pyspark import SparkContext ,SparkConf 

conf = SparkConf().setAppName("MyApp").setMaster("local")
sc = SparkContext(conf=conf)


