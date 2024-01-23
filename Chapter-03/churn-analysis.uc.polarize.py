# Databricks notebook source
# MAGIC %md
# MAGIC * [**Customer Churn**](https://en.wikipedia.org/wiki/Customer_attrition) also known as Customer attrition, customer turnover, or customer defection, is the loss of clients or customers and is...

# COMMAND ----------

# MAGIC %md
# MAGIC - https://docs.databricks.com/en/machine-learning/feature-store/example-notebooks.html

# COMMAND ----------

# MAGIC %pip install polars
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# read more about reading files from Databricks repos at https://docs.databricks.com/repos.html#access-files-in-a-repo-programmatically
# import os

import pyspark.pandas as ps
import numpy as np
import polars as pl

# COMMAND ----------

# from get_spark import GetSpark

# spark = GetSpark("field-eng-west").init_spark()
print(spark.conf.get("spark.databricks.workspaceUrl"))
print(spark.conf.get("spark.databricks.clusterUsageTags.clusterId"))

# COMMAND ----------

bank_df = (
    spark.read.option("header", True)
    .option("inferSchema", True)
    .csv("/Volumes/juan_dev/mldbxbook/mldbxbook/churn.csv")
    # .csv(f"file:{os.getcwd()}/data/churn.csv")
)

display(bank_df)

bank_df.select("Surname").distinct().count()

# COMMAND ----------

DATABASE_NAME = "mldbxbook"
CATALOG_NAME = "juan_dev"

# Create a new catalog with:
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG_NAME}")
spark.sql(f"USE CATALOG {CATALOG_NAME}")

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {DATABASE_NAME}")
spark.sql(f"USE SCHEMA {DATABASE_NAME}")

# COMMAND ----------

spark.sql("select current_database(), current_catalog()").show()

# COMMAND ----------

bank_df.write.format("delta").mode("overwrite").saveAsTable(f"{DATABASE_NAME}.raw_data")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Defining a feature engineering function that will return a Spark dataframe with a unique primary key.
# MAGIC In our case it is the `CustomerId`.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC The `bank_df` DataFrame is already pretty clean, but we do have some nominal features that we'll need to convert to numeric features for modeling.
# MAGIC
# MAGIC These features include:
# MAGIC
# MAGIC * **`Geography`**
# MAGIC * **`Gender`**
# MAGIC
# MAGIC We will also be dropping few features which dont add additional value for our model:
# MAGIC * **`RowNumber`**
# MAGIC * **`Surname`**
# MAGIC
# MAGIC ### Create `compute_features` Function
# MAGIC
# MAGIC A lot of data scientists are familiar with Pandas DataFrames, so we'll use the [pyspark.pandas](https://spark.apache.org/docs/3.2.0/api/python/user_guide/pandas_on_spark/) library to one-hot encode these categorical features.
# MAGIC
# MAGIC **Note:** we are creating a function to perform these computations. We'll use it to refer to this set of instructions when creating our feature table.

# COMMAND ----------


def compute_features(spark_df):
    # use polars instead of pandas on spark
    pl_ohe_df = (
        pl.from_pandas(spark_df.toPandas())
        .drop(["RowNumber", "Surname"])
        .to_dummies(columns=["Geography", "Gender"], drop_first=True)
    ).cast(
        {
            "Gender_Male": pl.Int32,
            "Geography_Germany": pl.Int32,
            "Geography_Spain": pl.Int32,
        }
    )

    return pl_ohe_df



# COMMAND ----------

# MAGIC %md
# MAGIC ### Compute Features
# MAGIC
# MAGIC Next, we can use our featurization function `compute_features` to create create a DataFrame of our features.

# COMMAND ----------


bank_features_df = compute_features(bank_df)
display(bank_features_df)

# COMMAND ----------

features_ohe_df = spark.createDataFrame(bank_features_df.to_pandas())

# COMMAND ----------

# Our first step is to instantiate the feature store client using `FeatureStoreClient()`.
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

fs = FeatureEngineeringClient()

# COMMAND ----------

bank_feature_table = fs.create_table(
    name=f"juan_dev.{DATABASE_NAME}.bank_customer_features",  # the name of the feature table
    primary_keys=["CustomerId"],  # primary key that will be used to perform joins
    schema=features_ohe_df.schema,  # the schema of the Feature table
    description="This customer level table contains one-hot encoded categorical and scaled numeric features to predict bank customer churn.",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Populate the feature table using write_table.
# MAGIC Now, we can write the records from **`bank_features_df`** to the feature table.

# COMMAND ----------

fs.write_table(
    df=features_ohe_df,
    name=f"{CATALOG_NAME}.{DATABASE_NAME}.bank_customer_features",
    mode="merge",
)
# instead of overwrite you can choose "merge" as an option if you want to update only certain records.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cleanup

# COMMAND ----------

# Drop feature table. This will drop the underlying Delta table as well.

fs.drop_table(
  name=f"{CATALOG_NAME}.{DATABASE_NAME}.bank_customer_features"
)

# COMMAND ----------

# MAGIC %md
# MAGIC Note: <b>In you decide to drop table from UI follow the follwing steps.</b>.
# MAGIC
# MAGIC Follow the following steps:
# MAGIC - Go to [Feature Store](/#feature-store/feature-store)
# MAGIC - Select the feature tables and select `delete` after clicking on 3 vertical dots icon.
# MAGIC
# MAGIC Deleting the feature tables in this way requires you to manually delete the published online tables and the underlying Delta table separately.
