# power_consumption_spark


##  Project Description

This project analyzes electricity consumption data using Apache Spark. The goal is to preprocess the data, extract relevant features (such as temperature, season, holiday, etc.), and build regression models to predict electricity usage. The project demonstrates how to use Spark DataFrame operations, feature engineering, and machine learning (MLlib) in a scalable pipeline.


##  Technologies Used
- Python 3.10
- Apache Spark 3.5.1 (PySpark)
- Spark MLlib
- Windows CMD (for batch file execution)


##  Dataset Description

The dataset contains daily electricity consumption records with weather and time-based features. Key columns include:

- `DATE`: The date of the record.
- `ENERGY`: Total energy consumed on the day (target variable).
- `T2M`, `T2M_MIN`, `T2M_MAX`: Daily average, minimum, and maximum temperatures.
- `RH2M`: Relative humidity at 2 meters.
- `ALLSKY`: Cloud coverage.
- `HDD18_3`, `CDD0`, `CDD10`: Heating/Cooling degree days (temperature-based indicators).
- `HOLIDAY`: Whether the day is a public holiday.

###  Extracted and Engineered Features

- `season`, `month`, `day_of_week`, `year`: Extracted temporal features from the date column.
- `temp_diff`: Difference between maximum and minimum temperature of the day.
- `temp_avg`: Average temperature calculated from `T2M_MIN` and `T2M_MAX`.
- `ENERGY_lag1`, `ENERGY_lag2`: Lag features representing energy consumption 1 and 2 days before.
- `energy_ma_3`: 3-day moving average of energy consumption.
- `energy_std_3`: 3-day rolling standard deviation of energy consumption.

The dataset is publicly available and was downloaded from [Mendeley Data](https://data.mendeley.com/datasets/mzkyh37mtr/2).

##  Scripts Overview

This folder contains the executable files of the project, consisting of 4 Python scripts. The execution order is as follows:
1. `preprocessing.py`
2. `feature_engineering.py`
3. `model_training.py`
4. `predict_power_consumption.py`

Each script is explained in its respective section below.

### `preprocessing.py`
- Loads and merges raw CSV data.
- Handles missing values and basic transformations.
- Removes duplicate rows.
- Detects and removes outliers from the dataset.

### `feature_engineering.py`
- Adds new columns:
  - `temp_diff`, `temp_avg`
  - `ENERGY_lag1`, `ENERGY_lag2`
  - `energy_ma_3`, `energy_std_3`
- Extracts time features: day_of_week, season, etc.

### `model_training.py`
- Trains two different regression models using Spark MLlib.
- Both models are trained using a pipeline that includes feature assembling and scaling.
- The `GBTRegressor` (Gradient-Boosted Trees Regressor) is used as the regression model in both cases.
- Each model is saved in a separate folder (`Model_Spark_v1/` and `Model_Spark_v2/`) to reflect different versions of training.

### `predict_power_consumption.py`
- Loads trained model.
- Makes predictions on new or test data.


##  Installation

All required packages and dependencies for running this project are listed in the `requirements.txt` file.

To set up and run the project:

1. Clone the repository:
```bash
git clone https://github.com/MOHAMMADSABOUR210/power_consumption_spark.git
cd power_consumption_spark

python -m venv venv
venv\Scripts\activate    # On Windows
source venv/bin/activate # On Linux/macOS

pip install -r requirements.txt
```

## How to Run

After installing the required dependencies, simply run the `run_spark.bat` file from the command line while inside the project folder.
```bash
run_spark.bat
```
Inside this batch file, each PySpark script is called separately. You can decide which scripts to run by commenting or uncommenting the lines with `@REM`. For example:

```bash
REM Run your PySpark script
@REM %PYSPARK_PYTHON% scripts/preprocessing.py
@REM %PYSPARK_PYTHON% scripts/feature_engineering.py
@REM %PYSPARK_PYTHON% scripts/model_training.py
%PYSPARK_PYTHON% scripts/predict_power_consumption.py
```

In this example, only `predict_power_consumption.py` will run when you execute `run_spark.bat`. 

Modify these lines to control which scripts get executed.



## License

This project is licensed under the MIT License.  
You are free to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software, subject to the conditions stated in the LICENSE file.  

The software is provided "as is", without any warranty of any kind, express or implied.  
For full details, please refer to the [LICENSE](LICENSE) file.

© 2025 Mohammad Sabour
