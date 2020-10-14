// Databricks notebook source
//import libraries 
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.functions._

// COMMAND ----------

// MAGIC %md Data Overview
// MAGIC -----------------

// COMMAND ----------

// DBTITLE 0,Data Overview
// read data
val df = table("compensation_data_csv")
df.show(5)

// COMMAND ----------

// drop empty columns c16 - c25
val df1 = df.drop("_c16","_c17","_c18","_c19","_c20","_c21","_c22","_c23","_c24","_c25")

// COMMAND ----------

// show schema
df1.printSchema()

// COMMAND ----------

// show number of rows
df1.count()

// COMMAND ----------

// MAGIC %md Data Cleaning
// MAGIC --------------------

// COMMAND ----------

// drop "Date" column as collected records are in the same time period
// drop "Level" column due to inconsistent format 
// drop "Base Salary", "Stock Grant", and "Bonus" columns since our target variable is total yearly compensation and these columns are not useful for our model development
val df2 = df1.drop("Date","Level","Base Salary (/year)","Stock Grant (/year)","Bonus (/year)")

// COMMAND ----------

// create an in memory reference to df2
df2.createTempView("df_t1")

// COMMAND ----------

// MAGIC %python
// MAGIC # import relevant libraries
// MAGIC import numpy as np
// MAGIC import pandas as pd
// MAGIC import matplotlib.pyplot as plt
// MAGIC import seaborn as sns
// MAGIC import pyspark.sql.functions as F
// MAGIC 
// MAGIC df_temp = sqlContext.table("df_t1")
// MAGIC 
// MAGIC # create heatmap of null values in dataset
// MAGIC h = sns.heatmap(df_temp.toPandas().isnull(), cmap='coolwarm', yticklabels=False, cbar=False)

// COMMAND ----------

// show number of null values in each column 
df2.select(df2.columns.map(c => sum(col(c).isNull.cast("int")).alias(c)): _*).show()

// COMMAND ----------

// drop null values in Total Yearly Compensation 
val df3 = df2.filter(df2.col("Total Yearly Compensation").isNotNull)

// show number of null values in each column 
df3.select(df3.columns.map(c => sum(col(c).isNull.cast("int")).alias(c)): _*).show()

// COMMAND ----------

// count of each unique value in Standard Level column
df3.groupBy($"Standard Level").agg(count("Standard Level")).orderBy($"count(Standard Level)".desc).show(false)

// COMMAND ----------

// replace nulls from standard level column with mode value "Software Engineer" 
val col1 = Array("Standard Level") 
val df4a = df3.na.fill("Software Engineer", col1)

// COMMAND ----------

// Skill Index is a numerical feature, replace Nulls with column mean

val col2 = Array("Skill Index") 

//important  - declare dataframe as var 
var df4b = df4a

//convert all columns to Double
for (col2 <- col2 ){df4b = df4b.withColumn(col2, col(col2).cast("Double"))}

//replace Nulls with mean using imputer function           
import org.apache.spark.ml.feature.Imputer

val imputer = new Imputer().
setInputCols(col2).
setOutputCols(col2.map(c => s"${c}_imputed")).
setStrategy("mean")

val df4c = imputer.fit(df4b).transform(df4b).drop(col2 : _*)
df4c.select("Skill Index_imputed").describe().show()

// COMMAND ----------

// replace Nulls in Gender feature with the string "Unknown" 
val col3 = Array("Gender") 
val df4d = df4c.na.fill("Unknown", col3)

// count of each unique value in Gender column
df4d.groupBy($"Gender").agg(count("Gender")).orderBy($"count(Gender)".desc).show(false)

// COMMAND ----------

// Other Details features where Null to be replaced with a cat value "Bachelors" 
val col4 = Array("Other Details") 
val df4e = df4d.na.fill("Bachelors", col4)
df4e.select("Other Details").distinct().count()

// COMMAND ----------

// create an in memory reference to df3
df4e.createTempView("df_t2")

// COMMAND ----------

// MAGIC %python
// MAGIC df_temp = sqlContext.table("df_t2")
// MAGIC new_df = df_temp.select(F.col("Total Yearly Compensation").cast("int"))
// MAGIC 
// MAGIC # distribution of total yearly compensation
// MAGIC fig, ax = plt.subplots(figsize=(8,5))
// MAGIC sns.distplot(new_df.select("Total Yearly Compensation").toPandas(), ax=ax)
// MAGIC plt.ylabel("Density")
// MAGIC plt.xlabel("Total Yearly Compensation")
// MAGIC plt.show()

// COMMAND ----------

// drop records with total yearly compensation over 1 million USD 
val df5 = df4e.filter(df4e.col("Total Yearly Compensation") < 1000000)

// describe target variable
df5.select("Total Yearly Compensation").describe().show()

// COMMAND ----------

df5.show(5)

// COMMAND ----------

// check number of null values in each column
df5.select(df5.columns.map(c => sum(col(c).isNull.cast("int")).alias(c)): _*).show()

// COMMAND ----------

// MAGIC %md Feature Engineering
// MAGIC -----------------------

// COMMAND ----------

//Define a Scala function to categorize Company into major companies and "Other"
val categorize_comp = udf { (x: String) => 
  if   (x == "Amazon")  "Amazon"
  
  else if   (x == "Deloitte")  "Deloitte"
  
  else if   (x == "KPMG")  "KPMG"
  
  else if (x=="EY") "EY"
  
  else if (x=="PwC") "PwC"
  
  else if (x=="McKinsey") "McKinsey"
  
  else if (x=="Accenture") "Accenture"
  
  else if (x=="Boeing") "Boeing"
  
  else if (x=="Facebook") "Facebook"
  
  else if (x=="LinkedIn") "LinkedIn"
  
  else if (x=="Twitter") "Twitter"
  
  else if (x=="Indeed") "Indeed"
  
  else if (x=="T-Mobile") "T-Mobile"
  
  else if (x=="AT&T") "AT&T"
  
  else if (x=="Bell") "Bell"
  
  else if (x=="Telus") "Telus"
  
  else if (x=="Ericsson") "Ericsson"
  
  else if (x=="IBM") "IBM"
  
  else if (x=="Microsoft") "Microsoft"
  
  else if (x=="Apple") "Apple"
  
  else if (x=="Salesforce") "Salesforce"
  
  else if (x=="Google") "Google"
  
  else if (x=="Cisco") "Cisco"
  
  else if (x=="Yahoo") "Yahoo"
  
  else if (x=="Walmart") "Walmart"
  
  else if (x=="General Motors") "General Motors"
  
  else if (x=="Samsung") "Samsung"
  
  else if (x=="Tesla") "Tesla"
  
  else if (x=="Dell") "Dell"
  
  else if (x=="Honeywell") "Honeywell"
  
  else if (x=="Siemens") "Siemens"
  
  else if (x=="Nike") "Nike"
  
  else if (x=="Intel") "Intel"
  
  else if (x ==  "JPMorgan Chase") "JPMorgan Chase"
  
  else if (x=="Visa") "Visa"
  
  else if (x=="Paypal") "Paypal"
  
  else if (x=="Bloomberg") "Bloomberg"
  
  else "Other"

}

// add categorized_company column to the dataframe
val df_1 = df5.withColumn("Cat_comp" , categorize_comp($"Company"))
df_1.show(5)

// COMMAND ----------

// count of each unique value in Title column
df_1.groupBy($"Title").agg(count("Title")).orderBy($"count(Title)".desc).show(300, false)

// COMMAND ----------

// define a Scala function to categorize Title
val categorize_title = udf{ (x:String) =>
  //1-Software Engineers
   if (x =="Software Engineer" || x =="Backend Developer" || x =="Application Developer" || x =="Applications Engineer" || x =="Sr Platform Analyst" || x =="Structures Engineer" || x =="Systems Development Engineer" || x =="Software Developer" || x =="Chase Associate Program" || x =="Software Testing")
   "Software Engineers"
    
  //2-Product Managers
   else if (x == "Product Manager" || x =="Associate Director" || x =="Director, IT" || x =="Advisor Program Manager" || x =="BIE" || x =="Senior Product Manager" || x =="Lead Administrator" || x =="Senior Director" || x =="Director of Sales")
   "Product Managers"
  
  //3-Engineering Manager
  else if (x =="Software Engineering Manager" || x =="Senior Computer Vision Software Engineer" || x =="Senior Principal Engineer" || x =="System Engineer" || x =="Solutions Manager" || x =="Lead Software Engineer" || x =="Techinical Leader" || x =="Hardware Engineering Manager" || x =="Director of IT" || x =="Senior Director") 
    "Engineering Manager"
  
  //4-Data 
  else if (x == "Data Scientist" || x =="Data Engineer, Data" || x =="Decision Scientist" || x =="BSA" || x =="Migration Engineer" || x =="Data Analyst" || x =="Sr. Data Engineer" || x =="DATA ANALYST") 
    "Data"
  
  //5-Seniors and Consultants
  else if (x == "Management Consultant" || x =="Senior QA Engineer" || x =="Senior data scientist" || x =="Solution Consultant")
   "Seniors and Consultants"
  
  //6-Hardware Engineer
  else if (x == "Hardware Engineer" || x =="ASIC design 3" || x =="Solutions Architect")
   "Hardware Engineer"
  
   //7-Product Designer
  else if (x == "Product Designer" || x =="Product Design Manager" || x =="Designer" || x =="Front End Developer" || x =="UX Architect" || x =="Conversational Designer" || x =="User Experience Researcher" || x =="Customer Engineer" || x =="Customer Success Manager" || x =="Product Specialist" || x =="Design Engineer")
   "Product Designer"
  
  //8-Technical
  else if (x == "Technical Program Manager" || x =="Member of Technical Staff" || x =="Network Engineer" || x =="Technical Account Manager" || x =="Technical Service Engineer" || x =="technical support engineer")
   "Technical"
  
  //9-Solution Architect and support 
  else if (x == "Solution Architect" || x =="Support engineer" || x =="Cloud Support Engineer" || x =="Distinguished Engineer")
   "Solution Architect and support"
  
  //10-Management and strategists
  else if (x == "Project Manager" || x =="Investment Banker" || x =="Digital Strategist" || x =="Manager – Technology" || x =="System Administrator" || x =="Sr. Manager Corporate Strategy" || x =="Associate Principal" || x =="Business Development" || x =="Finance, Manager" || x =="Portfolio Manager" || x =="Quantitative Analyst" || x =="Business systems analyst" || x =="Information Security Manager" || x =="Operations Manager")
   "Management and strategists"
  
  //11-Sales and marketing
  else if (x == "Marketing Operations" || x =="Financial Analyst" || x =="Market Operations Specialist" || x =="Marketing Specialist" || x =="Marketing Manager" || x =="Solution Sales Specialist" || x =="Technical Sales" || x =="Business" || x =="QE")
   "Sales and marketing"
  
  //12-HR 
  else if (x == "Staff Production Planner" || x =="Technical Recruiter" || x =="Staffing Services Associate")
   "HR"
  
  //13-Other titles 
  else if (x == "Civil Engineer" || x =="Quality Assurance Engineer" || x =="Applied Scientist" || x =="Fellow" || x =="PhD internship" || x =="Electrical Engineer" || x =="Security Engineer")
   "Other titles"
  
  else
  "Other"
}

// add general title column to the dataframe
val df_2 = df_1.withColumn("Cat_title" , categorize_title($"Title"))
df_2.show(5)

// COMMAND ----------

//Define a Scala function to categorize US locations
val categorizeUSLocation = udf { (x: String) => 
  if   (x == "WA" || x ==  "OR"|| x ==  "ID" || x ==  "MT" || x ==  "WY" || x ==   "CA" || x ==   "NV" || x ==  "UT" || x ==   "CO" || x == "AK" || x == "HI")  "US_West"
  
  else if (x == "ND" || x =="SD" || x =="NE" || x =="KS" || x =="MN" || x =="IA" || x =="MO" || x =="WI" || x =="IL" || x =="MI" || x =="IN" || x =="OH") "US_Midwest"
  
  else if (x == "PA" || x ==  "NJ" || x ==  "NY" || x ==  "CT" || x ==  "RI" || x ==   "MA" || x ==   "NH" || x ==  "VT" || x ==   "ME")  "US_Northeast"
  
  else if (x == "WV" || x == "MD" || x ==  "DC" || x ==  "DE" || x ==  "VA" || x ==  "KY" || x ==   "NC" || x ==   "TN" || x ==  "AR" || x ==   "LA" || x ==  "MS" || x ==  "AL" || x ==  "GA" || x ==  "SC" || x ==  "FL") "US_Southeast"
  
  else "US_Southwest"
  
}

//Define a Scala function to categorize other locations
val categorizeLocation = udf { (x: String) => 
  if   (x == "Canada")  "Canada"

  else if (x == "China" || x ==  "Korea" || x ==  "Singapore" || x ==   "Japan") "Asia1"
  
  else if (x== "India" || x== "Indonesia") "Asia2"
  
  else if (x == "Germany" || x == "Ireland" || x == "United Kingdom" || x == "Netherlands" ||  x == "Switzerland" ||  x == "Sweden" || x == "Luxembourg" ) "Europe1"
  
  else if (x == "Poland" || x == "Romania" || x == "Russia" ||x == "Bulgaria" ||x == "Belarus" || x == "Portugal" || x == "Finland") "Europe2"
  
  else "Other"

}

// split string by comma
val split_col = split($"Location", ", ")

// add categorized_location column to dataframe depending on number of split items
val df_3 = df_2.withColumn("Cat_location", when(size(split_col) < 3, categorizeUSLocation(split_col.getItem(1))).otherwise(categorizeLocation(split_col.getItem(2))))
df_3.show(5)

// COMMAND ----------

// add education column to dataframe
val degree = List("Masters","PhD","Masters and PhD")
val df_4 = df_3.withColumn("Masterorabove", col("Other Details").rlike(degree.mkString("|")))
df_4.show(5)

// COMMAND ----------

// count of each unique value in Tag column
df_4.groupBy($"Tag").agg(count("Tag")).orderBy($"count(Tag)".desc).show(300, false)

// COMMAND ----------

//Define a Scala function to categorize Tag using top 13 unique values with highest count and "Other"
val categorize_tag = udf{ (x:String) =>
  //Full Stack
   if (x =="Full Stack")
   "Full Stack"
    
  //Distributed Systems (Back-End)
   else if (x == "Distributed Systems (Back-End)")
   "Distributed Systems (Back-End)"
  
  //API Development (Back-End)
  else if (x =="API Development (Back-End)") 
    "API Development (Back-End)"
  
  //Web Development (Front-End)
  else if (x == "Web Development (Front-End)" ) 
    "Web Development (Front-End)"
  
  //ML / AI
  else if (x == "ML / AI")
   "ML / AI"
  
  //Reliability 
  else if (x == "Site Reliability (SRE)")
   "Site Reliability (SRE)"
  
  else if (x == "Networking")
  "Networking"
  
  else if (x =="iOS")
  "iOS"
  
  //Mobile (iOS + Android) 
  else if (x == "Mobile (iOS + Android)")
   "Mobile (iOS + Android)"
  
  //DevOps 
  else if (x == "DevOps")
   "DevOps"
  
  else if (x=="Security")
  "Security"
  
  else if (x=="Android")
  "Android"
  
  else if (x=="Testing (SDET)")
  "Testing (SDET)"

  else "Other"
  
}

// add categorized_tag column to dataframe
val df_5 = df_4.withColumn("Cat_tag", categorize_tag($"Tag"))
df_5.show(5)

// COMMAND ----------

val distinct_df = df_5.select(countDistinct("Cat_comp").as("count(DISTINCT Company)"),
                             countDistinct("Cat_title").as("count(DISTINCT Title)"),
                             countDistinct("Cat_location").as("count(DISTINCT Location)"),
                             countDistinct("Cat_tag").as("count(DISTINCT Tag)"))
distinct_df.show(false)

// COMMAND ----------

// import relevant libraries for one hot encoding
import org.apache.spark.ml.feature.{StringIndexer,OneHotEncoderEstimator}

//set an instance of StringIndexer
val indexer1 = new StringIndexer()
  .setInputCol("Cat_comp")
  .setOutputCol("Cat_comp_indexed")
  .setStringOrderType("frequencyAsc")

// fit StringIndexer and apply it on the dataset
val df_6 = indexer1.fit(df_5).transform(df_5)
df_6.show(5)

// COMMAND ----------

//apply string indexer to categorized title column
//set an instance of StringIndexer
val indexer2 = new StringIndexer()
  .setInputCol("Cat_title")
  .setOutputCol("Cat_title_indexed")
  .setStringOrderType("frequencyAsc")

//Fit StringIndexer and apply it on the dataset
val df_7 = indexer2.fit(df_6).transform(df_6)

val encoder2 = new OneHotEncoderEstimator()
  .setInputCols(Array("Cat_title_indexed"))
  .setOutputCols(Array("Cat_title_sparse_vec"))
  .setDropLast(false)

// transform the dataset by adding a new column with OneHotEncoder values
val df_8 = encoder2.fit(df_7).transform(df_7)

// convert sparse vector into dense vector 
import org.apache.spark.ml.linalg.{Vector, Vectors}

val sparsetoDense2  = udf((v:Vector) => v.toDense)

val df9 = df_8.withColumn("Cat_title_dense_vec", sparsetoDense2($"Cat_title_sparse_vec"))
df9.select("Cat_title_sparse_vec","Cat_title_dense_vec").show(5)

// COMMAND ----------

//apply onehotencoder to gender column 
//Step1: convert original categorocal values into mumerical categorical with StringIndexer
val indexer3 = new StringIndexer()
  .setInputCol("Gender")
  .setOutputCol("Gender_indexed")
  .setStringOrderType("frequencyAsc")

//Fit StringIndexer and apply it on the dataset
val df12 = indexer3.fit(df9).transform(df9)

//Step 2: conver numerical index value into a sparse vector using OneHotEncoderEstimator

val encoder3 = new OneHotEncoderEstimator()
  .setInputCols(Array("Gender_indexed"))
  .setOutputCols(Array("Gender_sparse_vec"))
  .setDropLast(false)

//lets transform the dataset by adding a new column with OneHotEncoder values
val df13 = encoder3.fit(df12).transform(df12)

// COMMAND ----------

//convert gender sparse vector into dense vector
val df14 = df13.withColumn("Gender_dense_vec", sparsetoDense2($"Gender_sparse_vec"))
df14.select("Gender_sparse_vec","Gender_dense_vec").show(5)

// COMMAND ----------

//apply onehotencoder to categorized location column 
//Step1: convert original categorocal values into mumerical categorical with StringIndexer
val indexer4 = new StringIndexer()
  .setInputCol("Cat_location")
  .setOutputCol("Cat_location_indexed")
  .setStringOrderType("frequencyAsc")

//Fit StringIndexer and apply it on the dataset
val df15 = indexer4.fit(df14).transform(df14)

//Step 2: conver numerical index value into a sparse vector using OneHotEncoderEstimator

val encoder4 = new OneHotEncoderEstimator()
  .setInputCols(Array("Cat_location_indexed"))
  .setOutputCols(Array("Location_sparse_vec"))
  .setDropLast(false)

//lets transform the dataset by adding a new column with OneHotEncoder values
val df16 = encoder4.fit(df15).transform(df15)

// COMMAND ----------

//convert location sparse vector into dense vector
val df17 = df16.withColumn("Location_dense_vec", sparsetoDense2($"Location_sparse_vec"))
df17.select("Location_sparse_vec","Location_dense_vec").show(5)

// COMMAND ----------

// apply stringindexer on categorized tag feature
val indexer5 = new StringIndexer()
  .setInputCol("Cat_tag")
  .setOutputCol("Cat_tag_Indexed")
  .setStringOrderType("frequencyAsc")

//Fit StringIndexer and apply it on the dataset salary
val df18 = indexer5.fit(df17).transform(df17)

// COMMAND ----------

//apply onehotencoder to standard level column 
//Step1: convert original categorocal values into mumerical categorical with StringIndexer
val indexer6 = new StringIndexer()
  .setInputCol("Standard Level")
  .setOutputCol("Standard Level Indexed")
  .setStringOrderType("frequencyAsc")

//Fit StringIndexer and apply it on the dataset salary
val df19 = indexer6.fit(df18).transform(df18)

//Step 2: conver numerical index value into a sparse vector using OneHotEncoderEstimator

val encoder6 = new OneHotEncoderEstimator()
  .setInputCols(Array("Standard Level Indexed"))
  .setOutputCols(Array("Standard_level_sparse_vec"))
  .setDropLast(false)

//lets transform the dataset by adding a new column with OneHotEncoder values
val df20 = encoder6.fit(df19).transform(df19)

// COMMAND ----------

//convert standard level sparse vector into dense vector
val df21 = df20.withColumn("Standard_level_dense_vec", sparsetoDense2($"Standard_level_sparse_vec"))
df21.select("Standard_level_sparse_vec","Standard_level_dense_vec").show(5)

// COMMAND ----------

// drop columns of original features
val df22 = df21.drop("Company","Title","Location","Tag","Gender","Other Details","Industry","General Title","Cat_location","Cat_tag","Cat_comp","Cat_title","Cat_title_indexed","Cat_title_sparse_vec","Gender_indexed","Gender_sparse_vec","Cat_location_indexed","Location_sparse_vec","Standard Level","Standard Level Indexed","Standard_level_sparse_vec")

// COMMAND ----------

//convert data to double for features that are supposed to be in numeric type but were casted as strings 

//define columns that have to be excluded from the conversion 
val exclude = Array("Skill Index_imputed","Masterorabove","Cat_comp_indexed", "Cat_title_dense_vec", "Gender_dense_vec","Location_dense_vec","Cat_tag_Indexed","Standard_level_dense_vec")
val df23 = (df22.columns.toBuffer --= exclude).foldLeft(df22)((current, c) => current.withColumn(c, col(c).cast("double")))

// COMMAND ----------

df23.describe().show(false)

// COMMAND ----------

import org.apache.spark.ml.feature.VectorAssembler

// columns that need to added to feature column
val cols = Array("Location_dense_vec", "Masterorabove", "Cat_comp_indexed", "Years of Experience", "Years at Company", "Cat_title_dense_vec", "Gender_dense_vec","Cat_tag_Indexed","Skill Index_imputed","Standard_level_dense_vec")

// VectorAssembler to add feature column
// input columns - cols
// feature column - features
val assembler = new VectorAssembler()
  .setInputCols(cols)
  .setOutputCol("features")

val featureDf = assembler.transform(df23)

//Split into training and test data
val Array(training, test) = featureDf.randomSplit(Array(0.7, 0.3), seed = 1)

// COMMAND ----------

// import relevant libraries for regression models
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit, CrossValidator}
import org.apache.spark.ml.evaluation.{RegressionEvaluator}
import org.apache.spark.mllib.evaluation.RegressionMetrics

// linear regression
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.mllib.regression.LinearRegressionWithSGD

// decision tree
import org.apache.spark.ml.regression.{DecisionTreeRegressionModel, DecisionTreeRegressor}

// random forest
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}

// gradient boosting
import org.apache.spark.ml.regression.{GBTRegressor, GBTRegressionModel} 

// COMMAND ----------

// MAGIC %md Linear Regression
// MAGIC ---------------------

// COMMAND ----------

// Define model to use
val lrmod = new LinearRegression()
  .setLabelCol("Total Yearly Compensation")
  .setFeaturesCol("features")

//Train the model on the training data
val initmodel = lrmod.fit(training)

// COMMAND ----------

// apply trained model on test data
val inittest = initmodel.transform(test).select("prediction", "Total Yearly Compensation")
inittest.show(5)

// COMMAND ----------

// Symmetric mean absolute percentage error Function
val SMAPE = udf{ (x:Double ,y:Double) =>
  (100*(((x-y).abs)/((x.abs)+ (y.abs))))
}

// mean absolute percentage error function
val MAPE = udf{ (x:Double ,y:Double) =>
  (100*(((x-y).abs)/((x.abs))))
}

// COMMAND ----------

// apply udf to find SMAPE
val smi = inittest.withColumn("smape" , SMAPE($"Total Yearly Compensation",$"prediction"))
val sumSmapei =  smi.select(avg($"smape")).first.get(0)
println(s"Symmetric mean absolute percentage error (SMAPE)  = $sumSmapei")

// COMMAND ----------

// apply udf to find MAPE
val mmi = inittest.withColumn("mape" , MAPE($"Total Yearly Compensation",$"prediction"))
val sumMapei =  mmi.select(avg($"mape")).first.get(0)
println(s"Mean absolute percentage error (MAPE)  = $sumMapei")

// COMMAND ----------

// Define model to use
val lr = new LinearRegression()
  .setLabelCol("Total Yearly Compensation")
  .setFeaturesCol("features")

// Define Parameters to go over
// Regularization Parameter
// Fitting Intercept
// Elastic Net

val paramGrid = new ParamGridBuilder()
  .addGrid(lr.regParam, Array(1.0, 0.5, 0.1, 0.05, 0.01))
  .addGrid(lr.fitIntercept)
  .addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0))
  .build()

// train validation split
val tvs = new TrainValidationSplit()
  .setEstimator(lr)
  .setEvaluator(new RegressionEvaluator()
                .setLabelCol("Total Yearly Compensation")
                .setPredictionCol("prediction"))
  .setEstimatorParamMaps(paramGrid)
  .setTrainRatio(0.75)

// train model on training data
val model = tvs.fit(training)

// apply trained model on test data
val holdout = model.transform(test).select("prediction", "Total Yearly Compensation")

val rm = new RegressionMetrics(holdout.rdd.map(x =>
  (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double])))
println("RMSE: " + rm.rootMeanSquaredError)
println("R Squared: " + rm.r2)

// COMMAND ----------

// SMAPE and MAPE evaluation
val sm = holdout.withColumn("smape" , SMAPE($"Total Yearly Compensation",$"prediction"))
val mm = holdout.withColumn("mape" , MAPE($"Total Yearly Compensation",$"prediction"))
val sumSmape =  sm.select(avg($"smape")).first.get(0)
println(s"Symmetric mean absolute percentage error (SMAPE)  = $sumSmape")
val sumMape =  mm.select(avg($"mape")).first.get(0)
println(s"Mean absolute percentage error (MAPE)  = $sumMape")

// COMMAND ----------

// declare an evaluator
val evaluator = new RegressionEvaluator()
.setLabelCol("Total Yearly Compensation")
.setPredictionCol("prediction")

//Set up a K-fold Cross Validator to go through the models and parameters assigned
val cv = new CrossValidator()
.setEstimator(lr)
.setEvaluator(evaluator)
.setEstimatorParamMaps(paramGrid)
.setNumFolds(5)

//Fit the training data to the model
val cvModel = cv.fit(training)

//Test cross validated model with test data
val cvPredictionDf = cvModel.transform(test)
cvPredictionDf.select("Total Yearly Compensation","prediction").show(5)

// COMMAND ----------

// SMAPE evaluation
val smcv = cvPredictionDf.withColumn("smape" , SMAPE($"Total Yearly Compensation",$"prediction"))
val sumSmapecv =  smcv.select(avg($"smape")).first.get(0)
println(s"Symmetric mean absolute percentage error (SMAPE)  = $sumSmapecv")

// MAPE evaluation
val mmcv = cvPredictionDf.withColumn("mape" , MAPE($"Total Yearly Compensation",$"prediction"))
val sumMapecv =  mmcv.select(avg($"mape")).first.get(0)
println(s"Mean absolute percentage error (MAPE)  = $sumMapecv")

// COMMAND ----------

// get parameters of best model
val bestModelLR = cvModel.bestModel.asInstanceOf[LinearRegressionModel].extractParamMap()

// COMMAND ----------

// MAGIC %md Decision Trees
// MAGIC ------------------

// COMMAND ----------

/// define DecisionTreeRegressor model.
val dt = new DecisionTreeRegressor()
  .setLabelCol("Total Yearly Compensation")
  .setFeaturesCol("features")
  .setMaxBins(37)

// train model
val dtModel = dt.fit(training)

// make predictions
val dtPredictions = dtModel.transform(test)

dtPredictions.select("Total Yearly Compensation", "prediction").show(5)

// COMMAND ----------

// apply udf to find SMAPE
val dt_smape_init = dtPredictions.withColumn("smape" , SMAPE($"Total Yearly Compensation",$"prediction"))
val sumSmapeDT =  dt_smape_init.select(avg($"smape")).first.get(0)
println(s"Symmetric mean absolute percentage error (SMAPE)  = $sumSmapeDT")

// COMMAND ----------

// building a paramGrid based on dt
// two hyperparameters are varied
val paramGridDT = new ParamGridBuilder()  
  .addGrid(dt.maxDepth, Array(2, 4, 5, 6, 7, 8, 9, 10)) // , 11, 12))
  .addGrid(dt.maxBins, Array(37, 40, 80))//, 200, 400, 800))
  .build()

//Set up a K-fold Cross Validator to go through the models and parameters assigned
val cvDT = new CrossValidator()
.setEstimator(dt)
.setEvaluator(evaluator)
.setEstimatorParamMaps(paramGridDT)
.setNumFolds(5)

// train model
val cvDTmodel = cvDT.fit(training)

// make predictions
val cvDTpredictions = cvDTmodel.transform(test)

// Select example rows to display.
cvDTpredictions.select("Total Yearly Compensation", "prediction").show(5)

// COMMAND ----------

// apply udf to find SMAPE
val dt_smape_fin = cvDTpredictions.withColumn("smape" , SMAPE($"Total Yearly Compensation",$"prediction"))
val sumSmapeDTcv =  dt_smape_fin.select(avg($"smape")).first.get(0)
println(s"Symmetric mean absolute percentage error (SMAPE)  = $sumSmapeDTcv")

// COMMAND ----------

// get parameters of best model
val bestModelDT = cvDTmodel.bestModel.asInstanceOf[DecisionTreeRegressionModel].extractParamMap()

// COMMAND ----------

// MAGIC %md Random Forest
// MAGIC -----------------

// COMMAND ----------

// Define a RandomForest model.
val rf = new RandomForestRegressor()
  .setLabelCol("Total Yearly Compensation")
  .setFeaturesCol("features")
  .setMaxBins(37)

// Train model. 
val model_rf = rf.fit(training)

// Make predictions.
val predictions_rf = model_rf.transform(test)

// Select example rows to display.
predictions_rf.select("Total Yearly Compensation", "prediction").show(5)

// COMMAND ----------

// apply udf to evaluate SMAPE
val predictions_smape_rf = predictions_rf.withColumn("smape" , SMAPE($"Total Yearly Compensation",$"prediction") )

val sumSmapeRF =  predictions_smape_rf.select(avg($"smape")).first.get(0)
println(s"Symmetric mean absolute percentage error (SMAPE)  = $sumSmapeRF%")

// COMMAND ----------

// parameters that needs to be tuned, we tune
//  1. num of trees
//  2. max depth

val paramGridRF = new ParamGridBuilder()
  .addGrid(rf.numTrees, Array(10, 20, 30))
  .addGrid(rf.maxDepth, Array(4, 6, 8, 10))
  .build()

// define cross validation stage to search through the parameters
// 5-Fold cross validation
val cvRF = new CrossValidator()
  .setEstimator(rf)
  .setEvaluator(evaluator)
  .setEstimatorParamMaps(paramGridRF)
  .setNumFolds(5)

// fit will run cross validation and choose the best set of parameters
// this will take some time to run
val cvModelRF = cvRF.fit(training)

// test cross validated model with test data
val cvPredictionsRF = cvModelRF.transform(test)
cvPredictionDf.select("Total Yearly Compensation", "prediction").show(5)

// COMMAND ----------

// measure the SMAPE of cross validated model
// this model is more accurate than the old model
val cv_predictions_smape = cvPredictionsRF.withColumn("smape", SMAPE($"Total Yearly Compensation", $"prediction") )

val sumSmapeRFcv =  cv_predictions_smape.select(avg($"smape")).first.get(0)
println(s"Symmetric mean absolute percentage error with cross validation(SMAPE)  = $sumSmapeRFcv%")

// COMMAND ----------

// get parameters of best model
val bestModelRF = cvModelRF.bestModel.asInstanceOf[RandomForestRegressionModel].extractParamMap()

// COMMAND ----------

// MAGIC %md Gradient Boosting
// MAGIC ---------------------

// COMMAND ----------

// Define GBT model
val gbt = new GBTRegressor()
  .setLabelCol("Total Yearly Compensation")
  .setFeaturesCol("features")
  .setMaxBins(37)
       
// Train model
val modelGBT = gbt.fit(training)

// Make predictions
val predictionsGBT = modelGBT.transform(test)

// compare target with prediction
predictionsGBT.select("Total Yearly Compensation", "prediction").show(5)

// COMMAND ----------

// Apply UDF to calculate SMAPE on testData
val smapeGBT = predictionsGBT.withColumn("smape", SMAPE($"Total Yearly Compensation", $"prediction"))
val avg_smape = smapeGBT.agg(avg($"smape"))
avg_smape.show()

// COMMAND ----------

// Define the grid of hyperparameters to test 
val paramGridGBT = new ParamGridBuilder()
  .addGrid(gbt.maxIter, Array(10, 20))
  .addGrid(gbt.maxDepth, Array(4, 7, 10))
  .addGrid(gbt.stepSize, Array(0.3, 0.1, 0.05, 0.01))
  .build()
       
// 5-fold cross-validation for hyperparameter tuning
val cvGBT = new CrossValidator() 
  .setEstimator(gbt) 
  .setEvaluator(evaluator) 
  .setEstimatorParamMaps(paramGridGBT) 
  .setNumFolds(5) 

// Takes very long!
// Train model
val cvModelGBT = cvGBT.fit(training)

// Make predictions
val cvPredictionsGBT = cvModelGBT.transform(test)
cvPredictionsGBT.select("Total Yearly Compensation", "prediction").show(5)

// COMMAND ----------

// Apply UDF to calculate SMAPE on testData
val cvSMAPEgbt = cvPredictionsGBT.withColumn("smape", SMAPE($"Total Yearly Compensation", $"prediction"))
val avg_cvSMAPE = cvSMAPEgbt.agg(avg($"smape"))
avg_cvSMAPE.show()

// COMMAND ----------

// get parameters of best model
val bestModelGBT = cvModelGBT.bestModel.asInstanceOf[GBTRegressionModel].extractParamMap()

// COMMAND ----------

// MAGIC %md Feature Importance
// MAGIC ----------------------

// COMMAND ----------

// Train a RandomForest model.
val rf = new RandomForestRegressor()
  .setLabelCol("Total Yearly Compensation")
  .setFeaturesCol("features")
  .setMaxBins(37)

// Train model. This also runs the indexer.
val model_rf = rf.fit(training)

// Make predictions.
val predictions = model_rf.transform(test)

// Select example rows to display.
predictions.select("prediction", "Total Yearly Compensation", "features").show(5)

val rfModel = model_rf.asInstanceOf[RandomForestRegressionModel]

//using SMAPE to evalute model error 
val SMAPE = udf{ (x:Double ,y:Double) =>
  (100*((x-y).abs)/((x.abs)+ (y.abs)))
}

val predictions_smape = predictions.withColumn("smape" , SMAPE($"Total Yearly Compensation",$"prediction") )

val sumSteps =  predictions_smape.select(avg($"smape")).first.get(0)
println(s"Symmetric mean absolute percentage error (SMAPE)  = $sumSteps%")

// COMMAND ----------

val df15_1 = df15.select("Cat_location","Cat_location_indexed").withColumnRenamed("Cat_location_indexed","Categorized_Location_Indexed")
val df15_new = df15_1.select("Cat_location","Categorized_Location_Indexed").withColumn("Index",expr("Categorized_Location_Indexed")).distinct

val df9_new = df9.select("Cat_title","Cat_title_indexed").withColumn("Index",expr("Cat_title_indexed + 15")).distinct

val df12_new = df12.select("Gender","Gender_indexed").withColumn("Index",expr("Gender_indexed + 29")).distinct

val df19_1 = df19.select("Standard Level","Standard Level indexed").withColumnRenamed("Standard Level indexed","Standard_Level_indexed")
val df19_new = df19_1.select("Standard Level","Standard_Level_indexed").withColumn("Index",expr("Standard_Level_indexed + 35")).distinct

// COMMAND ----------

import spark.implicits._
val combined_features = df15_new.union(df19_new).union(df9_new).union(df12_new).drop("Categorized_Location_Indexed")

// COMMAND ----------

val other_features = spark.createDataFrame(Seq(
  ("Masterorabove", 11.0),
  ("Cat_comp_indexed", 12.0),
  ("Years of Experience", 13.0),
  ("Years at Company", 14.0),
  ("Cat_tag_Indexed", 33.0),
  ("Skill Index_imputed", 34.0)
   
)).toDF("Feature_Name", "Index")


// COMMAND ----------

val Indices = other_features.union(combined_features).orderBy($"Index")
Indices.show(50)

// COMMAND ----------

val feature_idx = Indices.collect().map(_(0)).toArray

// COMMAND ----------

//get the feature importance from random forest model
val res = rfModel.featureImportances
val fi = feature_idx.zip(res.toArray).sortBy(-_._2)

// COMMAND ----------

//get the feature index of the 5 highest ranking
for (i <- 1 to 10) {
  print(s"Rank $i: ")
  println(fi(i-1))
}

// COMMAND ----------

//Feature importance according to the Pearson correlation

//include the target in VectorAssembler
val cols_with_target = Array("Location_dense_vec", "Masterorabove", "Cat_comp_indexed", "Years of Experience", "Years at Company", "Cat_title_dense_vec", "Gender_dense_vec","Cat_tag_Indexed","Skill Index_imputed","Standard_level_dense_vec","Total Yearly Compensation")
val assembler2 = new VectorAssembler()
  .setInputCols(cols_with_target)
  .setOutputCol("features_with_target")

val totalDf = assembler2.transform(df23)

// COMMAND ----------

import org.apache.spark.ml.linalg.{Matrix, Vectors}
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.sql.Row

//compute the Pearson correlation matrix
val Row(coeff: Matrix) = Correlation.corr(totalDf, "features_with_target").head
println(s"Pearson correlation matrix:\n $coeff")

// COMMAND ----------

val num = coeff.numCols
val corr = Array.fill(num)(0d)  
for (i <- 0 to (num-1)) {
      corr(i) = coeff(41,i)
}

// COMMAND ----------

//sort the correlation
val corr_abs = corr.map(x=>x.abs)
val sorted_corr = corr.sorted(Ordering.Double.reverse)
val sorted_corr_abs = corr_abs.sorted(Ordering.Double.reverse)

// COMMAND ----------

//Find the features with 5 highest ranking
val rank_index = Array.fill(num)(0)  
for (i <- 0 to 41) {
  //println(i,corr_abs.indexOf(sorted_corr(i)))
  rank_index(i) = corr_abs.indexOf(sorted_corr_abs(i))
}

// COMMAND ----------

val corr_zip = feature_idx.zip(corr)

// COMMAND ----------

//get the feature index of the 5 highest ranking
for (i <- 1 to 10) {
  print(s"Rank $i: ")
  println(corr_zip(rank_index(i)))

}
