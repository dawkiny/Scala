//###############################################################
//pyspark: export SPARK_MAJOR_VERSION=2 | pyspark --master local[2] --packages com.databricks:spark-csv_2.10:1.4.0
//spark-shell(scala): export SPARK_MAJOR_VERSION=2 | spark-shell --master local[2] --packages com.databricks:spark-csv_2.10:1.4.0
//###############################################################

// breast-cancer
val data2 = MLUtils.loadLibSVMFile(sc, "file:/usr/hdp/2.5.0.0-1245/spark2/data/mllib/breast-cancer_scale")
val data = data2.map{lp => var labelp = None: Option[LabeledPoint]; if (lp.label == 2.0) labelp = Some(LabeledPoint(0.0, lp.features))	else if (lp.label == 4.0) labelp = Some(LabeledPoint(1.0, lp.features)); labelp.get }


//val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")


// Adult (a9a)
val data2 = MLUtils.loadLibSVMFile(sc, "file:/usr/hdp/2.5.0.0-1245/spark2/data/mllib/a9a")
val data = data2.map{lp => var labelp = None: Option[LabeledPoint]; if (lp.label == -1.0) labelp = Some(LabeledPoint(0.0, lp.features))	else if (lp.label == +1.0) labelp = Some(LabeledPoint(1.0, lp.features)); labelp.get }


// Split data into training (60%) and test (40%).
val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
val training = splits(0).cache()
val test = splits(1)


// SVM
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
val svmAlg = new SVMWithSGD()
//svmAlg.optimizer.setNumIterations(200).setRegParam(0.1).setUpdater(new L1Updater)
svmAlg.optimizer.setNumIterations(100).setRegParam(1.0)
val model1 = svmAlg.run(training)


// Logistic Regression
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
val model2 = new LogisticRegressionWithLBFGS().setNumClasses(2).run(training)


// Decision Tree
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
val numClasses = 2
val categoricalFeaturesInfo = Map[Int, Int]()
val impurity = "gini"
val maxDepth = 5
val maxBins = 32
val model3 = DecisionTree.trainClassifier(training, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)


// Naive Bayes
//import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
//val model = NaiveBayes.train(training, lambda = 1.0, modelType = "multinomial")

//////////////////////////////////////////////
// Compute raw scores on the test set.
val scoreAndLabels1 = test.map { point => 
val score1 = model1.predict(point.features)
 (score1, point.label)
}
/////////////////////////////////////////////
//////////////////////////////////////////////
// Compute raw scores on the test set.
val scoreAndLabels2 = test.map { point => 
val score2 = model2.predict(point.features)
 (score2, point.label)
}
/////////////////////////////////////////////
//////////////////////////////////////////////
// Compute raw scores on the test set.
val scoreAndLabels3 = test.map { point => 
val score3 = model3.predict(point.features)
 (score3, point.label)
}
/////////////////////////////////////////////

/////////////////////////////////////////////
// Get evaluation metrics.
val metrics1 = new BinaryClassificationMetrics(scoreAndLabels1)
val auROC1 = metrics1.areaUnderROC()

println("Area under ROC = " + auROC1)
// Get evaluation metrics.
val metrics1 = new MulticlassMetrics(scoreAndLabels1)
val precision1 = metrics1.precision
println("Precision = " + precision1)
/////////////////////////////////////////////

/////////////////////////////////////////////
// Get evaluation metrics.
val metrics2 = new BinaryClassificationMetrics(scoreAndLabels2)
val auROC2 = metrics2.areaUnderROC()

println("Area under ROC = " + auROC2)

// Get evaluation metrics.
val metrics2 = new MulticlassMetrics(scoreAndLabels2)
val precision2 = metrics2.precision
println("Precision = " + precision2)
/////////////////////////////////////////////

/////////////////////////////////////////////
// Get evaluation metrics.
val metrics3 = new BinaryClassificationMetrics(scoreAndLabels3)
val auROC3 = metrics3.areaUnderROC()

println("Area under ROC = " + auROC3)

// Get evaluation metrics.
val metrics3 = new MulticlassMetrics(scoreAndLabels3)
val precision3 = metrics3.precision
println("Precision = " + precision3)
/////////////////////////////////////////////


/************************************ DataFrame  실습 ******************************************/
///in console:  wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data



import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.Row

val training = spark.createDataFrame(Seq(
  (1.0, Vectors.dense(0.0, 1.1, 0.1)),
  (0.0, Vectors.dense(2.0, 1.0, -1.0)),
  (0.0, Vectors.dense(2.0, 1.3, 1.0)),
  (1.0, Vectors.dense(0.0, 1.2, -0.5))
)).toDF("label", "features")
training.count
training.show


val df = spark.read.json("file:/usr/hdp/2.5.0.0-1245/spark2/examples/src/main/resources/people.json")
df.count
df.show

// val adult_df = spark.read.json()
// df.count
// df.show


// spark.csv library 이용
import org.apache.spark.sql.SQLContext
import spark.implicits._
val sqlContext = new SQLContext(sc)
val df = sqlContext.read.format("com.databricks.spark.csv").option("header", "false").option("inferSchema", "true").load("file:/usr/hdp/2.5.0.0-1245/spark2/data/mllib/adult.data")
df.show
df.printSchema

// rename column names
val df2 = df.withColumnRenamed("_c0", "age").withColumnRenamed("_c1","workclass").withColumnRenamed("_c2", "fnlwgt").withColumnRenamed("_c3","education").withColumnRenamed("_c4","education-num").withColumnRenamed("_c5","marital-status").withColumnRenamed("_c6","occupation").withColumnRenamed("_c7","relationship").withColumnRenamed("_c8","race").withColumnRenamed("_c9","sex").withColumnRenamed("_c10","capital-gain").withColumnRenamed("_c11","capital-loss").withColumnRenamed("_c12","hours-per-week").withColumnRenamed("_c13","native-country").withColumnRenamed("_c14","label")
df2.show
df2.printSchema


// select columns
val select_df = df2.select("education", "occupation")
select_df.show

// filter
val filter_df = df2.filter($"age" > 50)
filter_df.count
filter_df.show

// filter2
val filter2_df = df2.filter($"education-num" > 10)
filter2_df.count
filter2_df.show


// groupBy
val groupBy_df = df2.groupBy("marital-status").count
groupBy_df.show


// groupBy 후 내림차순 sorting
val groupBySort_df = df2.groupBy("marital-status").count.sort(desc("count"))
groupBySort_df.show


// null 혹은 empty space 찾기
df2.filter(df2("race").isNull || df2("race") === "").count


/************************************ Spark ML package 실습 ******************************************/

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.Row

// Prepare training data from a list of (label, features) tuples.
val training = spark.createDataFrame(Seq(
  (1.0, Vectors.dense(0.0, 1.1, 0.1)),
  (0.0, Vectors.dense(2.0, 1.0, -1.0)),
  (0.0, Vectors.dense(2.0, 1.3, 1.0)),
  (1.0, Vectors.dense(0.0, 1.2, -0.5))
)).toDF("label", "features")


// Create a LogisticRegression instance. This instance is an Estimator.
val lr = new LogisticRegression()
// Print out the parameters, documentation, and any default values.
println("LogisticRegression parameters:\n" + lr.explainParams() + "\n")


// We may set parameters using setter methods.
lr.setMaxIter(10).setRegParam(0.01)


// Learn a LogisticRegression model. This uses the parameters stored in lr.
val model1 = lr.fit(training)
// Since model1 is a Model (i.e., a Transformer produced by an Estimator),
// we can view the parameters it used during fit().
// This prints the parameter (name: value) pairs, where names are unique IDs for this
// LogisticRegression instance.
println("Model 1 was fit using parameters: " + model1.parent.extractParamMap)

// We may alternatively specify parameters using a ParamMap,
// which supports several methods for specifying parameters.
val paramMap = ParamMap(lr.maxIter -> 20).put(lr.maxIter, 30).put(lr.regParam -> 0.1, lr.threshold -> 0.55)

// One can also combine ParamMaps.
val paramMap2 = ParamMap(lr.probabilityCol -> "myProbability")  // Change output column name.
val paramMapCombined = paramMap ++ paramMap2

// Now learn a new model using the paramMapCombined parameters.
// paramMapCombined overrides all parameters set earlier via lr.set* methods.
val model2 = lr.fit(training, paramMapCombined)
println("Model 2 was fit using parameters: " + model2.parent.extractParamMap)

// Prepare test data.
val test = spark.createDataFrame(Seq(
  (1.0, Vectors.dense(-1.0, 1.5, 1.3)),
  (0.0, Vectors.dense(3.0, 2.0, -0.1)),
  (1.0, Vectors.dense(0.0, 2.2, -1.5))
)).toDF("label", "features")


// Make predictions on test data using the Transformer.transform() method.
// LogisticRegression.transform will only use the 'features' column.
// Note that model2.transform() outputs a 'myProbability' column instead of the usual
// 'probability' column since we renamed the lr.probabilityCol parameter previously.
model2.transform(test).select("features", "label", "myProbability", "prediction").collect().foreach {case Row(features: Vector, label: Double, prob: Vector, prediction: Double) =>println(s"($features, $label) -> prob=$prob, prediction=$prediction")}


// **************  Feature Extractor - Word2Vec

import org.apache.spark.ml.feature.Word2Vec

// Input data: Each row is a bag of words from a sentence or document.
val documentDF = spark.createDataFrame(Seq(
  "Hi I heard about Spark".split(" "),
  "I wish Java could use case classes".split(" "),
  "Logistic regression models are neat".split(" ")
).map(Tuple1.apply)).toDF("text")


// Learn a mapping from words to Vectors.
val word2Vec = new Word2Vec().setInputCol("text").setOutputCol("result").setVectorSize(3).setMinCount(0)
val model = word2Vec.fit(documentDF)
val result = model.transform(documentDF)
result.select("result").take(3).foreach(println)


// **************  Exercise: Feature Extractor - Word2Vec

import org.apache.spark.ml.feature.Word2Vec

// Input data: Each row is a bag of words from a sentence or document.
val documentDF2 = spark.createDataFrame(Seq(
  "Hi I like your bag".split(" "),
  "I really want to get it".split(" "),
  "Would you give me your bag".split(" "),
  "That was a joke hahaha".split(" ")
).map(Tuple1.apply)).toDF("text")


// Learn a mapping from words to Vectors.
val word2Vec = new Word2Vec().setInputCol("text").setOutputCol("result").setVectorSize(3).setMinCount(0)
val model2 = word2Vec.fit(documentDF2)
val result2 = model2.transform(documentDF2)
result2.select("result").take(4).foreach(println)


// **************  Feature Transformer - Tokenizer

import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer}

val sentenceDataFrame = spark.createDataFrame(Seq(
  (0, "Hi I heard about Spark"),
  (1, "I wish Java could use case classes"),
  (2, "Logistic,regression,models,are,neat")
)).toDF("label", "sentence")

val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")


val tokenized = tokenizer.transform(sentenceDataFrame)
tokenized.select("words", "label").take(3).foreach(println)

tokenized.show


// **************  Exercise: Feature Transformer - Tokenizer

import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer}

val sentenceDataFrame2 = spark.createDataFrame(Seq(
  (0, "Hi I like your bag"),
  (1, "I really want to get it"),
  (2, "Would you give me your bag"),
  (3, "That was a joke hahaha")
)).toDF("label", "sentence")

val tokenizer2 = new Tokenizer().setInputCol("sentence").setOutputCol("words")


val tokenized2 = tokenizer2.transform(sentenceDataFrame2)
tokenized2.select("words", "label").take(4).foreach(println)

tokenized2.show

// **************  Exercise2: Feature Transformer - Tokenizer(without label)
//
//import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer}
//
//val sentenceDataFrame3 = spark.createDataFrame(Seq(
//  "Hi I like your bag",
//  "I really want to get it",
//  "Would you give me your bag",
//  "That was a joke hahaha"
//), Seq(
//  "Hi I like your bag",
//  "I really want to get it",
//  "Would you give me your bag",
//  "That was a joke hahaha"
//)).toDF("sentence")
//
//val tokenizer3 = new Tokenizer().setInputCol("sentence").setOutputCol("words")
//
//
//val tokenized3 = tokenizer3.transform(sentenceDataFrame3)
//tokenized3.select("words", "label").take(4).foreach(println)
//
//tokenized3.show



// **************  Feature Transformer - PCA

import org.apache.spark.ml.feature.PCA
import org.apache.spark.ml.linalg.Vectors

val data = Array(
  Vectors.sparse(5, Seq((1, 1.0), (3, 7.0))),
  Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
  Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0)
)

val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")
val pca = new PCA().setInputCol("features").setOutputCol("pcaFeatures").setK(3).fit(df)
val pcaDF = pca.transform(df)
val result = pcaDF.select("pcaFeatures")
result.foreach(println(_))



// **************  Exercise: Feature Transformer - PCA

import org.apache.spark.ml.feature.PCA
import org.apache.spark.ml.linalg.Vectors

val data2 = Array(
  Vectors.dense(1.0, 8.0, 5.0, 12.0, 9.0, 2.0),
  Vectors.dense(3.0, 1.0, 4.0, 5.0, 9.0, 12.0),
  Vectors.dense(7.0, 2.0, 9.0, 11.0, 13.0, 3.0),
  Vectors.dense(4.0, 7.0, 1.0, 2.0, 11.0, 2.0),
  Vectors.dense(3.0, 6.0, 4.0, 3.0, 4.0, 6.0)
)

val df2 = spark.createDataFrame(data2.map(Tuple1.apply)).toDF("features")
val pca2 = new PCA().setInputCol("features").setOutputCol("pcaFeatures").setK(4).fit(df2)
val pcaDF2 = pca2.transform(df2)
val result2 = pcaDF2.select("pcaFeatures")
result2.foreach(println(_))


// **************  Feature Transformer - StringIndexer
//String 빈도 수에 따라 0부터 시작
import org.apache.spark.ml.feature.StringIndexer

val df = spark.createDataFrame(
  Seq((0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c"))
).toDF("id", "category")

val indexer = new StringIndexer().setInputCol("category").setOutputCol("categoryIndex")

val indexed = indexer.fit(df).transform(df)
indexed.show()


// **************  Exercise: Feature Transformer - StringIndexer

import org.apache.spark.ml.feature.StringIndexer

val df2 = spark.createDataFrame(
  Seq((0, "BMW"), (1, "Benz"), (2, "Hyundai"), (3, "Kia"), (4, "Toyota"), (5, "Kia"))
).toDF("id", "category")

val indexer2 = new StringIndexer().setInputCol("category").setOutputCol("categoryIndex")

val indexed2 = indexer2.fit(df2).transform(df2)
indexed2.show()



// **************  Feature Transformer - IndexToString


import org.apache.spark.ml.feature.{IndexToString, StringIndexer}

val df = spark.createDataFrame(Seq(
  (0, "a"),
  (1, "b"),
  (2, "c"),
  (3, "a"),
  (4, "a"),
  (5, "c")
)).toDF("id", "category")

val indexer = new StringIndexer().setInputCol("category").setOutputCol("categoryIndex").fit(df)
val indexed = indexer.transform(df)

indexed.show
val converter = new IndexToString().setInputCol("categoryIndex").setOutputCol("originalCategory")

val converted = converter.transform(indexed)
converted.select("id", "originalCategory").show()



// **************  Feature Transformer - VectorIndexer (Example)


import org.apache.spark.ml.feature.VectorIndexer

val data = spark.read.format("libsvm").load("file:/usr/hdp/2.5.0.0-1245/spark2/data/mllib/sample_libsvm_data.txt")

val indexer = new VectorIndexer().setInputCol("features").setOutputCol("indexed").setMaxCategories(10)

val indexerModel = indexer.fit(data)

val categoricalFeatures: Set[Int] = indexerModel.categoryMaps.keys.toSet
println(s"Chose ${categoricalFeatures.size} categorical features: " + categoricalFeatures.mkString(", "))

// Create new column "indexed" with categorical values transformed to indices
val indexedData = indexerModel.transform(data)
indexedData.show()



// **************  Feature Transformer - VectorIndexer

import org.apache.spark.ml.linalg.Vectors

val data = Array(
  Vectors.dense(1.0, 2.0, 3.0, 4.0, 5.0),
  Vectors.dense(1.0, 1.0, 2.0, 3.0, 4.0),
  Vectors.dense(1.0, 2.0, 1.0, 2.0, 3.0),
  Vectors.dense(1.0, 1.0, 3.0, 1.0, 2.0),
  Vectors.dense(1.0, 1.0, 2.0, 1.0, 1.0)
)

val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

val indexer = new VectorIndexer().setInputCol("features").setOutputCol("indexed").setMaxCategories(3)
val indexerModel = indexer.fit(df)

// Create new column "indexed" with categorical values transformed to indices
val indexedData = indexerModel.transform(df)
indexedData.take(5).foreach(println(_))



// **************  Feature Transformer - Normalizer

import org.apache.spark.ml.feature.Normalizer

val data = Array(
  Vectors.dense(1.0, 2.0, 3.0, 4.0, 5.0),
  Vectors.dense(1.0, 1.0, 2.0, 3.0, 4.0),
  Vectors.dense(1.0, 2.0, 1.0, 2.0, 3.0),
  Vectors.dense(1.0, 1.0, 3.0, 1.0, 2.0),
  Vectors.dense(1.0, 1.0, 2.0, 1.0, 1.0)
)
val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

// Normalize each Vector using $L^1$ norm.
val normalizer = new Normalizer().setInputCol("features").setOutputCol("normFeatures").setP(2.0)

val l1NormData = normalizer.transform(df)
l1NormData.take(5).foreach(println(_))



// **************  Exercise: Feature Transformer - Normalizer

import org.apache.spark.ml.feature.Normalizer

val data2 = Array(
  Vectors.dense(32.0, 15.0, 4.0, 21.0, 9.0),
  Vectors.dense(12.0, 78.0, 54.0, 33.0, 12.0),
  Vectors.dense(43.0, 11.0, 13.0, 96.0, 4.0)
)
val df2 = spark.createDataFrame(data2.map(Tuple1.apply)).toDF("features")

// Normalize each Vector using $L^1$ norm.
val normalizer2 = new Normalizer().setInputCol("features").setOutputCol("normFeatures").setP(2.0)

val l1NormData2 = normalizer2.transform(df2)
l1NormData2.take(5).foreach(println(_))




// **************  Feature Transformer - Vector Assembler

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

val dataset = spark.createDataFrame(
  Seq((0, 18, 1.0, Vectors.dense(0.0, 10.0, 0.5), 1.0))
).toDF("id", "hour", "mobile", "userFeatures", "clicked")

val assembler = new VectorAssembler().setInputCols(Array("hour", "mobile", "userFeatures")).setOutputCol("features")

val output = assembler.transform(dataset)
println(output.select("features").first())



// **************  Exercise: Feature Transformer - Vector Assembler

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

val dataset2 = spark.createDataFrame(
  Seq((8.3, 2.8, 3.1, 0, Vectors.dense(0, 3, 2), 4))
).toDF("c1", "c2", "c3", "c4", "c5", "c6")

val assembler2 = new VectorAssembler().setInputCols(Array("c1", "c3", "c5")).setOutputCol("features")

val output2 = assembler2.transform(dataset2)
println(output2.select("features").first())




// **************  Feature Transformer - QuantileDiscretizer


import org.apache.spark.ml.feature.QuantileDiscretizer

val data = Array((0, 18.0), (1, 19.0), (2, 8.0), (3, 5.0), (4, 2.2))
var df = spark.createDataFrame(data).toDF("id", "hour")

val discretizer = new QuantileDiscretizer().setInputCol("hour").setOutputCol("result").setNumBuckets(3)

val result = discretizer.fit(df).transform(df)
result.show()




// **************  Exercise: Feature Transformer - QuantileDiscretizer


import org.apache.spark.ml.feature.QuantileDiscretizer

val data2 = Array((0, 33.1), (1, 28.3), (2, 16.3), (3, -3.7), (4, -20.1))
var df2 = spark.createDataFrame(data2).toDF("id", "temp")

val discretizer2 = new QuantileDiscretizer().setInputCol("temp").setOutputCol("result").setNumBuckets(3)

val result2 = discretizer2.fit(df2).transform(df2)
result2.show()




// **************  Pipeline 


import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.Row

// Prepare training documents from a list of (id, text, label) tuples.
val training = spark.createDataFrame(Seq(
  (0L, "a b c d e spark", 1.0),
  (1L, "b d", 0.0),
  (2L, "spark f g h", 1.0),
  (3L, "hadoop mapreduce", 0.0)
)).toDF("id", "text", "label")

// Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
val hashingTF = new HashingTF().setNumFeatures(1000).setInputCol(tokenizer.getOutputCol).setOutputCol("features")
val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.01)
val pipeline = new Pipeline().setStages(Array(tokenizer, hashingTF, lr))

// Fit the pipeline to training documents.
val model = pipeline.fit(training)

// Prepare test documents, which are unlabeled (id, text) tuples.
val test = spark.createDataFrame(Seq(
  (4L, "spark i j k"),
  (5L, "l m n"),
  (6L, "mapreduce spark"),
  (7L, "apache hadoop")
)).toDF("id", "text")

// Make predictions on test documents.
model.transform(test).select("id", "text", "probability", "prediction").collect().foreach { case Row(id: Long, text: String, prob: Vector, prediction: Double) =>
    println(s"($id, $text) --> prob=$prob, prediction=$prediction")
}


// Now we can optionally save the fitted pipeline to disk
model.write.overwrite().save("file:/usr/hdp/2.5.0.0-1245/spark2/data/mllib_save/spark-logistic-regression-model")

// We can also save this unfit pipeline to disk
pipeline.write.overwrite().save("file:/usr/hdp/2.5.0.0-1245/spark2/data/mllib_save/unfit-lr-model")

// And load it back in during production
val sameModel = PipelineModel.load("file:/usr/hdp/2.5.0.0-1245/spark2/data/mllib_save/spark-logistic-regression-model")



// **************  Classification: Logistic Regression

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

// Load training data
val data = spark.read.format("libsvm").load("file:/usr/hdp/2.5.0.0-1245/spark2/data/mllib/sample_libsvm_data.txt")

val Array(training, test) = data.randomSplit(Array(0.7, 0.3))

val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)

// Fit the model
val lrModel = lr.fit(training)

// Print the coefficients and intercept for logistic regression
println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

val predictions = lrModel.transform(test)
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))


import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegression}

// Extract the summary from the returned LogisticRegressionModel instance trained in the earlier
// example
val trainingSummary = lrModel.summary

// Obtain the objective per iteration.
val objectiveHistory = trainingSummary.objectiveHistory
objectiveHistory.foreach(loss => println(loss))


// **************  Classification: Decision Tree


import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

// Load the data stored in LIBSVM format as a DataFrame.
val data = spark.read.format("libsvm").load("file:/usr/hdp/2.5.0.0-1245/spark2/data/mllib/sample_libsvm_data.txt")

// Index labels, adding metadata to the label column.
// Fit on whole dataset to include all labels in index.
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)
// Automatically identify categorical features, and index them.
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)

// Split the data into training and test sets (30% held out for testing).
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))


// Train a DecisionTree model.
val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")

// Convert indexed labels back to original labels.
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

// Chain indexers and tree in a Pipeline.
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

// Train model. This also runs the indexers.
val model = pipeline.fit(trainingData)

// Make predictions.
val predictions = model.transform(testData)

// Select example rows to display.
predictions.select("predictedLabel", "label", "features").show(5)

// Select (prediction, true label) and compute test error.
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))

val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
println("Learned classification tree model:\n" + treeModel.toDebugString)



// **************  Classification: Random Forest

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

// Load and parse the data file, converting it to a DataFrame.
val data = spark.read.format("libsvm").load("file:/usr/hdp/2.5.0.0-1245/spark2/data/mllib/sample_libsvm_data.txt")

// Index labels, adding metadata to the label column.
// Fit on whole dataset to include all labels in index.
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)

// Automatically identify categorical features, and index them.
// Set maxCategories so features with > 4 distinct values are treated as continuous.
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)

// Split the data into training and test sets (30% held out for testing).
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))


// Train a RandomForest model.
val rf = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(10)

// Convert indexed labels back to original labels.
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

// Chain indexers and forest in a Pipeline.
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

// Train model. This also runs the indexers.
val model = pipeline.fit(trainingData)

// Make predictions.
val predictions = model.transform(testData)

// Select example rows to display.
predictions.select("predictedLabel", "label", "features").show(5)

// Select (prediction, true label) and compute test error.
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))

val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
println("Learned classification forest model:\n" + rfModel.toDebugString)



// **************  Classification: Gradient Boosted Tree

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

// Load and parse the data file, converting it to a DataFrame.
val data = spark.read.format("libsvm").load("file:/usr/hdp/2.5.0.0-1245/spark2/data/mllib/sample_libsvm_data.txt")

// Index labels, adding metadata to the label column.
// Fit on whole dataset to include all labels in index.
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)
// Automatically identify categorical features, and index them.
// Set maxCategories so features with > 4 distinct values are treated as continuous.
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)

// Split the data into training and test sets (30% held out for testing).
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

// Train a GBT model.
val gbt = new GBTClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setMaxIter(10)

// Convert indexed labels back to original labels.
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

// Chain indexers and GBT in a Pipeline.
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, gbt, labelConverter))

// Train model. This also runs the indexers.
val model = pipeline.fit(trainingData)

// Make predictions.
val predictions = model.transform(testData)

// Select example rows to display.
predictions.select("predictedLabel", "label", "features").show(5)

// Select (prediction, true label) and compute test error.
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))

val gbtModel = model.stages(2).asInstanceOf[GBTClassificationModel]
println("Learned classification GBT model:\n" + gbtModel.toDebugString)



// **************  Classification: Naive Bayes

import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

// Load the data stored in LIBSVM format as a DataFrame.
val data = spark.read.format("libsvm").load("file:/usr/hdp/2.5.0.0-1245/spark2/data/mllib/sample_libsvm_data.txt")

// Split the data into training and test sets (30% held out for testing)
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

// Train a NaiveBayes model.
val model = new NaiveBayes().fit(trainingData)

// Select example rows to display.
val predictions = model.transform(testData)
predictions.show()

// Select (prediction, true label) and compute test error
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println("Accuracy: " + accuracy)


// **************  Regression: Linear Regressoion

import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.evaluation.RegressionEvaluator

// Load training data
val data = spark.read.format("libsvm").load("file:/usr/hdp/2.5.0.0-1245/spark2/data/mllib/sample_linear_regression_data.txt")

val lr = new LinearRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)

// Split the data into training and test sets (30% held out for testing)
val Array(training, test) = data.randomSplit(Array(0.7, 0.3))

// Fit the model
val lrModel = lr.fit(training)

val predictions = lrModel.transform(test)
predictions.show()


// Select (prediction, true label) and compute test error.
val evaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("rmse")
val rmse = evaluator.evaluate(predictions)
println("Root Mean Squared Error (RMSE) on test data = " + rmse)


// Print the coefficients and intercept for linear regression
println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

// Summarize the model over the training set and print out some metrics
val trainingSummary = lrModel.summary
println(s"numIterations: ${trainingSummary.totalIterations}")
println(s"objectiveHistory: ${trainingSummary.objectiveHistory.toList}")
trainingSummary.residuals.show()
println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
println(s"r2: ${trainingSummary.r2}")


// **************  Clustering: K-means

import org.apache.spark.ml.clustering.KMeans

// Loads data.
val dataset = spark.read.format("libsvm").load("file:/usr/hdp/2.5.0.0-1245/spark2/data/mllib/sample_kmeans_data.txt")

// Trains a k-means model.
val kmeans = new KMeans().setK(2).setSeed(1L)
val model = kmeans.fit(dataset)

// Evaluate clustering by computing Within Set Sum of Squared Errors.
val WSSSE = model.computeCost(dataset)
println(s"Within Set Sum of Squared Errors = $WSSSE")

// Shows the result.
println("Cluster Centers: ")
model.clusterCenters.foreach(println)

val presult = model.transform(dataset)
presult.show

/////////////////////////////////////////////////////
////sparkml_classification_creditcard_default.scala
/////////////////////////////////////////////////////


import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.Normalizer
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier, LogisticRegression, DecisionTreeClassifier, BinaryLogisticRegressionSummary}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer, VectorAssembler}
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator, TrainValidationSplit}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}


//------------------- Logistic Regression---------------------------//
val sqlContext = new org.apache.spark.sql.SQLContext(sc)


// csv파일에 header유무 확인, delimiter 확인
// Input file loading
val df = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load("file:/usr/hdp/2.5.0.0-1245/spark2/data/mllib/creditCardDefault.csv")

// schema 정보 및 data확인
df.printSchema

// 컬럼 개수 확인
df.schema.length

// 데이터 내용 흝어보기
df.show


// Training 과 Test 셋으로 Split Data (7:3비율)
val Array(trainingData, testData) = df.randomSplit(Array(0.7, 0.3), seed = 11L)


// Label 컬럼 및 Feature가 될만할 컬럼 파악
// 의미있는 컬럼만 추려서 Vector Assembler로 추린 컬럼들을 사용하여 feature컬럼 만들기
val columns = df.schema.map(a=>a.name).slice(1,24).toArray
val assembler = new VectorAssembler().setInputCols(columns).setOutputCol("features")

// LabelIndexer로 Label컬럼을 LabelIndex로 변경
val labelIndexer = new StringIndexer().setInputCol("default_payment_next_month").setOutputCol("indexedLabel")

//val encoder = new OneHotEncoder().setInputCol("indexedLabel2").setOutputCol("indexedLabel")

// FeatureIndexer
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(8)


// 알고리즘 취사선택
val algorithm = new LogisticRegression().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
//val algorithm = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
//val algorithm = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(10)
//val algorithm = new GBTClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setMaxIter(10)


// Label Converter
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.fit(trainingData).labels)

// Pipeline 만들기
val pipeline = new Pipeline().setStages(Array(assembler, labelIndexer, featureIndexer, algorithm, labelConverter))

// Training 
val model = pipeline.fit(trainingData)

// Test
val result = model.transform(testData)

// 결과 보기
result.printSchema
result.show


// Evaluation
val evaluator1 = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy1 = evaluator.evaluate(result)
println("Test Error = " + (1.0 - accuracy1))


// Logistic Regression Model Summary
// pipeline의 4번째 구성요소 접근
val alg_model = model.stages(3).asInstanceOf[LogisticRegressionModel]
val lr_summary = alg_model.summary
val objectiveHistory = lr_summary.objectiveHistory
objectiveHistory.foreach(loss => println(loss))


val binarySummary = lr_summary.asInstanceOf[BinaryLogisticRegressionSummary]
val roc = binarySummary.roc
roc.show()
println(binarySummary.areaUnderROC)


//------------------- Decision Tree---------------------------//
val sqlContext = new org.apache.spark.sql.SQLContext(sc)


// csv파일에 header유무 확인, delimiter 확인
// Input file loading
val df = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load("file:/usr/hdp/2.5.0.0-1245/spark2/data/mllib/creditCardDefault.csv")

// schema 정보 및 data확인
df.printSchema

// 컬럼 개수 확인
df.schema.length

// 데이터 내용 흝어보기
df.show


// Training 과 Test 셋으로 Split Data (7:3비율)
val Array(trainingData, testData) = df.randomSplit(Array(0.7, 0.3), seed = 11L)


// Label 컬럼 및 Feature가 될만할 컬럼 파악
// 의미있는 컬럼만 추려서 Vector Assembler로 추린 컬럼들을 사용하여 feature컬럼 만들기
val columns = df.schema.map(a=>a.name).slice(1,24).toArray
val assembler = new VectorAssembler().setInputCols(columns).setOutputCol("features")

// LabelIndexer로 Label컬럼을 LabelIndex로 변경
val labelIndexer = new StringIndexer().setInputCol("default_payment_next_month").setOutputCol("indexedLabel")

//val encoder = new OneHotEncoder().setInputCol("indexedLabel2").setOutputCol("indexedLabel")

// FeatureIndexer
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(8)


// 알고리즘 취사선택
//val algorithm = new LogisticRegression().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
val algorithm = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
//val algorithm = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(10)
//val algorithm = new GBTClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setMaxIter(10)


// Label Converter
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.fit(trainingData).labels)

// Pipeline 만들기
val pipeline = new Pipeline().setStages(Array(assembler, labelIndexer, featureIndexer, algorithm, labelConverter))

// Training 
val model = pipeline.fit(trainingData)

// Test
val result = model.transform(testData)

// 결과 보기
result.printSchema
result.show


// Evaluation
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy2 = evaluator.evaluate(result)
println("Test Error = " + (1.0 - accuracy2))


// Logistic Regression Model Summary
// pipeline의 4번째 구성요소 접근
val alg_model = model.stages(3).asInstanceOf[LogisticRegressionModel]
val lr_summary = alg_model.summary
val objectiveHistory = lr_summary.objectiveHistory
objectiveHistory.foreach(loss => println(loss))


val binarySummary = lr_summary.asInstanceOf[BinaryLogisticRegressionSummary]
val roc = binarySummary.roc
roc.show()
println(binarySummary.areaUnderROC)


//------------------- Random Forest---------------------------//
val sqlContext = new org.apache.spark.sql.SQLContext(sc)


// csv파일에 header유무 확인, delimiter 확인
// Input file loading
val df = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load("file:/usr/hdp/2.5.0.0-1245/spark2/data/mllib/creditCardDefault.csv")

// schema 정보 및 data확인
df.printSchema

// 컬럼 개수 확인
df.schema.length

// 데이터 내용 흝어보기
df.show


// Training 과 Test 셋으로 Split Data (7:3비율)
val Array(trainingData, testData) = df.randomSplit(Array(0.7, 0.3), seed = 11L)


// Label 컬럼 및 Feature가 될만할 컬럼 파악
// 의미있는 컬럼만 추려서 Vector Assembler로 추린 컬럼들을 사용하여 feature컬럼 만들기
val columns = df.schema.map(a=>a.name).slice(1,24).toArray
val assembler = new VectorAssembler().setInputCols(columns).setOutputCol("features")

// LabelIndexer로 Label컬럼을 LabelIndex로 변경
val labelIndexer = new StringIndexer().setInputCol("default_payment_next_month").setOutputCol("indexedLabel")

//val encoder = new OneHotEncoder().setInputCol("indexedLabel2").setOutputCol("indexedLabel")

// FeatureIndexer
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(8)


// 알고리즘 취사선택
//val algorithm = new LogisticRegression().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
//val algorithm = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
val algorithm = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(10)
//val algorithm = new GBTClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setMaxIter(10)


// Label Converter
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.fit(trainingData).labels)

// Pipeline 만들기
val pipeline = new Pipeline().setStages(Array(assembler, labelIndexer, featureIndexer, algorithm, labelConverter))

// Training 
val model = pipeline.fit(trainingData)

// Test
val result = model.transform(testData)

// 결과 보기
result.printSchema
result.show


// Evaluation
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy3 = evaluator.evaluate(result)
println("Test Error = " + (1.0 - accuracy3))


// Logistic Regression Model Summary
// pipeline의 4번째 구성요소 접근
val alg_model = model.stages(3).asInstanceOf[LogisticRegressionModel]
val lr_summary = alg_model.summary
val objectiveHistory = lr_summary.objectiveHistory
objectiveHistory.foreach(loss => println(loss))


val binarySummary = lr_summary.asInstanceOf[BinaryLogisticRegressionSummary]
val roc = binarySummary.roc
roc.show()
println(binarySummary.areaUnderROC)

//------------------- Gradient Boosted Tree---------------------------//
val sqlContext = new org.apache.spark.sql.SQLContext(sc)


// csv파일에 header유무 확인, delimiter 확인
// Input file loading
val df = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load("file:/usr/hdp/2.5.0.0-1245/spark2/data/mllib/creditCardDefault.csv")

// schema 정보 및 data확인
df.printSchema

// 컬럼 개수 확인
df.schema.length

// 데이터 내용 흝어보기
df.show


// Training 과 Test 셋으로 Split Data (7:3비율)
val Array(trainingData, testData) = df.randomSplit(Array(0.7, 0.3), seed = 11L)


// Label 컬럼 및 Feature가 될만할 컬럼 파악
// 의미있는 컬럼만 추려서 Vector Assembler로 추린 컬럼들을 사용하여 feature컬럼 만들기
val columns = df.schema.map(a=>a.name).slice(1,24).toArray
val assembler = new VectorAssembler().setInputCols(columns).setOutputCol("features")

// LabelIndexer로 Label컬럼을 LabelIndex로 변경
val labelIndexer = new StringIndexer().setInputCol("default_payment_next_month").setOutputCol("indexedLabel")

//val encoder = new OneHotEncoder().setInputCol("indexedLabel2").setOutputCol("indexedLabel")

// FeatureIndexer
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(8)


// 알고리즘 취사선택
//val algorithm = new LogisticRegression().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
//val algorithm = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
//val algorithm = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(10)
val algorithm = new GBTClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setMaxIter(10)


// Label Converter
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.fit(trainingData).labels)

// Pipeline 만들기
val pipeline = new Pipeline().setStages(Array(assembler, labelIndexer, featureIndexer, algorithm, labelConverter))

// Training 
val model = pipeline.fit(trainingData)

// Test
val result = model.transform(testData)

// 결과 보기
result.printSchema
result.show


// Evaluation
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy4 = evaluator.evaluate(result)
println("Test Error = " + (1.0 - accuracy4))


// Logistic Regression Model Summary
// pipeline의 4번째 구성요소 접근
val alg_model = model.stages(3).asInstanceOf[LogisticRegressionModel]
val lr_summary = alg_model.summary
val objectiveHistory = lr_summary.objectiveHistory
objectiveHistory.foreach(loss => println(loss))


val binarySummary = lr_summary.asInstanceOf[BinaryLogisticRegressionSummary]
val roc = binarySummary.roc
roc.show()
println(binarySummary.areaUnderROC)



// Compare Results
println("Test Error = " + (1.0 - accuracy1))
println("Test Error = " + (1.0 - accuracy2))
println("Test Error = " + (1.0 - accuracy3))
println("Test Error = " + (1.0 - accuracy4))



// ****************************** Cross-Validation ************************* //

val sqlContext = new org.apache.spark.sql.SQLContext(sc)
val df = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load("file:/usr/hdp/2.5.0.0-1245/spark2/data/mllib/creditCardDefault.csv")
val Array(trainingData, testData) = df.randomSplit(Array(0.7, 0.3), seed = 11L)

val columns = df.schema.map(a=>a.name).slice(1,24).toArray
val assembler = new VectorAssembler().setInputCols(columns).setOutputCol("features")
val labelIndexer = new StringIndexer().setInputCol("default_payment_next_month").setOutputCol("indexedLabel")
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4)
val algorithm = new LogisticRegression().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.fit(trainingData).labels)
val pipeline = new Pipeline().setStages(Array(assembler, labelIndexer, featureIndexer, algorithm, labelConverter))

// For cross validation
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val paramGrid = new ParamGridBuilder().addGrid(algorithm.regParam, Array(0.1, 0.01)).addGrid(algorithm.fitIntercept).addGrid(algorithm.elasticNetParam, Array(0.0, 0.5, 1.0)).build()
val cv = new CrossValidator().setEstimator(pipeline).setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setNumFolds(3)

// Training
val model = cv.fit(trainingData)

// Test
val result = model.transform(testData)

// Evaluation
val accuracy = evaluator.evaluate(result)
println("Test Error = " + (1.0 - accuracy))


///////////////////////////////////////
////sparkml_classification_iris.scala
///////////////////////////////////////


import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.Normalizer
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier, LogisticRegression, DecisionTreeClassifier, BinaryLogisticRegressionSummary}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer, VectorAssembler}
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator, TrainValidationSplit}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer, VectorAssembler}

val sqlContext = new org.apache.spark.sql.SQLContext(sc)
 

// csv파일에 header유무 확인, delimiter 확인
// Input file loading..
val df = sqlContext.read.format("com.databricks.spark.csv").option("header", "false").option("inferSchema", "true").load("file:/usr/hdp/2.5.0.0-1245/spark2/data/mllib/iris.data")

// schema 정보 및 data확인
df.printSchema

// 컬럼 개수 확인
df.schema.length

// 데이터 내용 흝어보기
df.show


// Training 과 Test 셋으로 Split Data (7:3비율)
val Array(trainingData, testData) = df.randomSplit(Array(0.7, 0.3))


// Label 컬럼 및 Feature가 될만할 컬럼 파악
// 의미있는 컬럼만 추려서 Vector Assembler로 추린 컬럼들을 사용하여 feature컬럼 만들기
val columns = df.schema.map(a=>a.name).slice(0,4).toArray
val assembler = new VectorAssembler().setInputCols(columns).setOutputCol("features")

// LabelIndexer로 Label컬럼을 LabelIndex로 변경
val labelIndexer = new StringIndexer().setInputCol("_c4").setOutputCol("indexedLabel")

// FeatureIndexer
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4)


// 알고리즘 취사선택
//val algorithm = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
val algorithm = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(10)


// Label Converter
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.fit(trainingData).labels)

// Pipeline 만들기
val pipeline = new Pipeline().setStages(Array(assembler, labelIndexer, featureIndexer, algorithm, labelConverter))

// Training 
val model = pipeline.fit(trainingData)

// Test
val result = model.transform(testData)

// 결과 보기
result.printSchema
result.show


// Evaluation
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(result)
println("Test Error = " + (1.0 - accuracy))



///////////////////////////////////////////
////sparkml_regression_bike_sharing.scala
///////////////////////////////////////////


import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.Normalizer
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer, VectorAssembler}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor, LinearRegression, GBTRegressionModel, GBTRegressor, DecisionTreeRegressor}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.types._
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator, TrainValidationSplit}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.Row
import org.apache.spark.ml.{Pipeline, PipelineModel}

val sqlContext = new org.apache.spark.sql.SQLContext(sc)


// csv파일에 header유무 확인, delimiter 확인
// Input file loading
val df = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load("file:/usr/hdp/2.5.0.0-1245/spark2/data/mllib/hour.csv")

// schema 정보 및 data확인
df.printSchema

// 컬럼 개수 확인
df.schema.length

// 데이터 내용 흝어보기
df.show


// Label 컬럼 및 Feature가 될만할 컬럼 파악 및 Label컬럼이 Int형이므로 Double로 바꾸어 주어야 한다.
val df2 = df.withColumn("cntTmp", df("cnt")cast(DoubleType)).drop("cnt").withColumnRenamed("cntTmp", "cnt")

// Training 과 Test 셋으로 Split Data (7:3비율)
val Array(trainingData, testData) = df.randomSplit(Array(0.7, 0.3), seed = 11L)


// 의미있는 컬럼만 추려서 Vector Assembler로 추린 컬럼들을 사용하여 feature컬럼 만들기
val columns = df.schema.map(a=>a.name).slice(2,14).toArray
val assembler = new VectorAssembler().setInputCols(columns).setOutputCol("features")

// FeatureIndexer
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(24)


// 알고리즘 취사선택
//val algorithm = new LinearRegression().setLabelCol("cnt").setFeaturesCol("indexedFeatures")
//val algorithm = new DecisionTreeRegressor().setLabelCol("cnt").setFeaturesCol("indexedFeatures")
//val algorithm = new RandomForestRegressor().setLabelCol("cnt").setFeaturesCol("indexedFeatures")
val algorithm = new GBTRegressor().setLabelCol("cnt").setFeaturesCol("indexedFeatures").setMaxIter(30)


// pipeline 생성
val pipeline = new Pipeline().setStages(Array(assembler, featureIndexer, algorithm))

// import org.apache.spark.ml.feature.VectorIndexerModel
// val vimodel = model.stages(1).asInstanceOf[VectorIndexerModel]
// val categoricalFeatures: Set[Int] = vimodel.categoryMaps.keys.toSet
// println(s"Chose ${categoricalFeatures.size} categorical features: " + categoricalFeatures.mkString(", "))


// Training 
val model = pipeline.fit(trainingData)

// Test
val result = model.transform(testData)

// 결과보기
result.show


// Evaluator Define
val evaluator = new RegressionEvaluator().setLabelCol("cnt").setPredictionCol("prediction").setMetricName("rmse")
val eval_result = evaluator.evaluate(result)
println("Root Mean Squared Error (RMSE) on test data = " + eval_result)



///////////////////////////////////////////
////sparkml_classification_titanic.scala
///////////////////////////////////////////



import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.Normalizer
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier, LogisticRegression, DecisionTreeClassifier, BinaryLogisticRegressionSummary}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer, VectorAssembler}
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator, TrainValidationSplit}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}


val sqlContext = new org.apache.spark.sql.SQLContext(sc)

// Input file loading
//val train = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load("file:/usr/hdp/2.5.0.0-1245/spark2/data/mllib/titanic_train.csv")
//val test = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load("file:/usr/hdp/2.5.0.0-1245/spark2/data/mllib/titanic_test.csv")

val df = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load("file:/usr/hdp/2.5.0.0-1245/spark2/data/mllib/titanic_train.csv")
val Array(train, test) = df.randomSplit(Array(0.7, 0.3), seed = 11L)


// Print Data Schema
train.printSchema
// |-- PassengerId: integer (nullable = true)
// |-- Survived: integer (nullable = true)
// |-- Pclass: integer (nullable = true)
// |-- Name: string (nullable = true)
// |-- Sex: string (nullable = true)
// |-- Age: double (nullable = true)
// |-- SibSp: integer (nullable = true)
// |-- Parch: integer (nullable = true)
// |-- Ticket: string (nullable = true)
// |-- Fare: double (nullable = true)
// |-- Cabin: string (nullable = true)
// |-- Embarked: string (nullable = true)


// na value remove if needed
//val train = train.na.drop()


// 전체 컬럼에 대해서 na 값 찾기
train.schema.map(a=>a.name).foreach{name=>println("Name: " + name); train.filter(train(name).isNull || train(name) === "").show}
// => Age, Cabin, Embarked 에 na 존재


// column count show
println("Total row count: " + train.count)
train.schema.map(a=>a.name).foreach{name=> train.groupBy(train(name)).count.sort(desc("count")).show}
// => 전체 row수는 891

// ********** Feature Selection ********** //
// Age : null count - 177 --> 평균값으로 대체 
// Cabin: "" count - 687  --> 전체 개수에 비해 너무 많아서 그냥 무시
// Embarked: "" count - 2 --> 가장 많은 빈도로 나타난 값으로 대체


// replace missing value with the average in Age column
val avg_age = train.select(mean("Age")).first()(0).asInstanceOf[Double]
val train2 = train.na.fill(avg_age, Seq("Age"))


// what about "" (empty string).. just use udf
val replaceEmpty = sqlContext.udf.register("replaceEmpty", (embarked: String) => {if (embarked  == "") "S" else embarked })
val train3 = train2.withColumn("Embarked", replaceEmpty(train2("Embarked")))

// check the result
train3.groupBy(train3("Embarked")).count.sort(desc("count")).show



// adding some useful features.. using user-defined function
val addChild = sqlContext.udf.register("addChild", (sex: String, age: Double) => {if (age < 15) "Child" else sex })
val withFamily = sqlContext.udf.register("withFamily", (sib: Int, par: Int) => {if (sib + par > 3) 1.0 else 0.0 })

val train4 = train3.withColumn("Sex", addChild(train3("Sex"), train3("Age")))
val train5 = train4.withColumn("Family", withFamily(train4("SibSp"), train4("Parch")))


// Check the schema
train5.printSchema
// |-- PassengerId: integer (nullable = true)
// |-- Survived: integer (nullable = true)
// |-- Pclass: integer (nullable = true)
// |-- Name: string (nullable = true)
// |-- Sex: string (nullable = true)
// |-- Age: double (nullable = false)
// |-- SibSp: integer (nullable = true)
// |-- Parch: integer (nullable = true)
// |-- Ticket: string (nullable = true)
// |-- Fare: double (nullable = true)
// |-- Cabin: string (nullable = true)
// |-- Embarked: string (nullable = true)
// |-- Family: double (nullable = true)


// Use StringIndexer on String schema type
val sexIndex = new StringIndexer().setInputCol("Sex").setOutputCol("SexIndex")
val embarkedIndex = new StringIndexer().setInputCol("Embarked").setOutputCol("EmbarkedIndex")
val survivedIndex = new StringIndexer().setInputCol("Survived").setOutputCol("SurvivedIndex")

//val encoder1 = new OneHotEncoder().setInputCol("SexIndex2").setOutputCol("SexIndex")
//val encoder2 = new OneHotEncoder().setInputCol("EmbarkedIndex2").setOutputCol("EmbarkedIndex")


// Select columns and Vector Assembler
val columns = Seq("Pclass", "SexIndex", "Age", "Fare", "EmbarkedIndex", "Family").toArray
val assembler = new VectorAssembler().setInputCols(columns).setOutputCol("features")

// Normalizer
//val normalizer = new Normalizer().setInputCol("features_temp").setOutputCol("features")

// FeatureIndexer
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(8)

// Classifiers
//val classmodel = new RandomForestClassifier().setLabelCol("SurvivedIndex").setFeaturesCol("indexedFeatures").setNumTrees(10)
//val classmodel = new GBTClassifier().setLabelCol("SurvivedIndex").setFeaturesCol("indexedFeatures").setMaxIter(10)
//val classmodel = new DecisionTreeClassifier().setLabelCol("SurvivedIndex").setFeaturesCol("indexedFeatures")
val classmodel = new LogisticRegression().setLabelCol("SurvivedIndex").setFeaturesCol("indexedFeatures")

// LabelConverter
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(survivedIndex.fit(train).labels)

// Pipeline 
//val pipeline = new Pipeline().setStages(Array(sexIndex, embarkedIndex, survivedIndex, encoder1, encoder2, assembler, normalizer, featureIndexer, classmodel, labelConverter))
//val pipeline = new Pipeline().setStages(Array(sexIndex, embarkedIndex, survivedIndex, assembler, normalizer, featureIndexer, classmodel, labelConverter))
val pipeline = new Pipeline().setStages(Array(sexIndex, embarkedIndex, survivedIndex, assembler, featureIndexer, classmodel, labelConverter))


// Training
//val model = pipeline.fit(train5)

// For cross validation
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("SurvivedIndex").setPredictionCol("prediction").setMetricName("accuracy")
val paramGrid = new ParamGridBuilder().addGrid(classmodel.regParam, Array(0.5, 0.1, 0.01)).addGrid(classmodel.fitIntercept).addGrid(classmodel.elasticNetParam, Array(0.0, 0.5, 1.0)).build()
val cv = new CrossValidator().setEstimator(pipeline).setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setNumFolds(3)

// Training
val model = cv.fit(train5)


// Test data pre-processig
val avg_age = test.select(mean("Age")).first()(0).asInstanceOf[Double]
val test2 = test.na.fill(avg_age, Seq("Age"))
val test3 = test2.withColumn("Sex", addChild(test2("Sex"), test2("Age")))
val test4 = test3.withColumn("Family", withFamily(test3("SibSp"), test3("Parch")))

println("Total row count: " + test4.count)
test4.schema.map(a=>a.name).foreach{name=> test4.groupBy(test4(name)).count.sort(desc("count")).show}


val getZero = sqlContext.udf.register("toDouble", ((n: Int) => { 0 }))
val test5 = test4.withColumn("Survived", getZero(test4("PassengerId")))
val test6 = test5.na.drop()


// Test 
val result = model.transform(test6)

//result.schema.map(a=>a.name).foreach{name=> result.groupBy(result(name)).count.sort(desc("count")).show}
//result.select("PassengerId", "predictedLabel").write.format("com.databricks.spark.csv").option("header", "true").save("data_result/titanic_result.csv")


//Evaluator 
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("SurvivedIndex").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(result)
println("Test Error = " + (1.0 - accuracy))



