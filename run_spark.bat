@echo off

REM Activate virtual environment
call D:\Programming\Data_Engineering\Apache_Spark\project\venv\Scripts\activate.bat

REM Set PySpark Python environment
set PYSPARK_PYTHON=D:\Programming\Data_Engineering\Apache_Spark\project\venv\Scripts\python.exe
set PYSPARK_DRIVER_PYTHON=D:\Programming\Data_Engineering\Apache_Spark\project\venv\Scripts\python.exe

REM Set Hadoop environment
set HADOOP_HOME=C:\hadoop\hadoop-3.3.6
set PATH=%HADOOP_HOME%\bin;%PATH%

REM Run your PySpark script
%PYSPARK_PYTHON% scripts/preprocessing.py

pause
