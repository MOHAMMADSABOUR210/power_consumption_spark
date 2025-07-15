@echo off

REM 
call D:\Programming\Data_Engineering\Apache_Spark\project\venv\Scripts\activate.bat

REM 
set PYSPARK_PYTHON=D:\Programming\Data_Engineering\Apache_Spark\project\venv\Scripts\python.exe
set PYSPARK_DRIVER_PYTHON=D:\Programming\Data_Engineering\Apache_Spark\project\venv\Scripts\python.exe

REM 
%PYSPARK_PYTHON% scripts/preprocessing.py


pause
