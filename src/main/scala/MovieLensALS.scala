import java.util.Random

import org.apache.log4j.Logger
import org.apache.log4j.Level

import scala.io.Source

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd._
import org.apache.spark.mllib.recommendation.{ALS, Rating, MatrixFactorizationModel}

// refer
// http://ampcamp.berkeley.edu/big-data-mini-course/movie-recommendation-with-mllib.html
// Version: access rate files from HDFS
// Wed May 20 06:30:36 PDT 2015
// Changed by Weiping He
// Tue May 26 14:59:55 PDT 2015, edited by Weiping. Train the data before asking for user input.

object MovieLensALS {

  def main(args: Array[String]) {

    // remove verbose log
    Logger.getLogger("org.apache.spark").setLevel(Level.OFF)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)    

    if (args.length != 1) {
      println("Usage: sbt/sbt package \"run movieLensHomeDir\"")
      exit(1)
    }

    // // args(0):  movieLensHomeDir, e.g. /user/hdfs
    /* [hdfs@localhost ~]$ hadoop fs -ls /user/hdfs
	-rw-r--r--   1 hdfs supergroup     171308 2015-05-14 18:18 /user/hdfs/movies.dat
	-rw-r--r--   1 hdfs supergroup   24594131 2015-05-15 09:17 /user/hdfs/ratings.dat
	-rw-r--r--   1 hdfs supergroup     134368 2015-05-14 18:18 /user/hdfs/users.dat
    */

    // To run it under spark :
    /*
     /usr/local/spark/bin/spark-submit --executor-memory 512m --master spark://localhost.localdomain:7077 --driver-memory 512m --class MovieLensALS /var/lib/hadoop-hdfs/testmlib/target/scala-2.10/movielens-als_2.10-0.1.jar /user/hdfs
    */

    // set up environment
    // sbt package will create ./target/scala-2.10/movielens-als_2.10-0.1.jar
    val jarFile = "target/scala-2.10/movielens-als_2.10-0.1.jar"

    //    val sparkHome = "/root/spark"
    //    val master = Source.fromFile("/root/spark-ec2/cluster-url").mkString.trim
    //    val masterHostname = Source.fromFile("/root/spark-ec2/masters").mkString.trim
    val sparkHome = "/usr/local/spark"
    val master ="spark://localhost.localdomain:7077"
    val masterHostname = "localhost.localdomain"

    val conf = new SparkConf()
      .setMaster(master)
      .setSparkHome(sparkHome)
      .setAppName("MovieLensALS")
      .set("spark.executor.memory", "512m")
      .setJars(Seq(jarFile))
    val sc = new SparkContext(conf)

    // load ratings and movie titles

    // hdfs://localhost.localdomain:8020/user/hdfs
    val movieLensHomeDir = "hdfs://" + masterHostname + ":8020"+ args(0)

    val ratings = sc.textFile(movieLensHomeDir + "/ratings.dat").map { line =>
      val fields = line.split("::")
      // UserID::MovieID::Rating::Timestamp
      // e.g.
      // 1::1193::5::978300760
      // The RDD contains (Int, Rating) pairs.
      // We only keep the last digit of the timestamp as a random key: = fields(3).toLong % 10
      // The Rating class is a wrapper around tuple (user: Int, product: Int, rating: Double) 
      //      defined in org.apache.spark.mllib.recommendation package.
      // format: (timestamp % 10, Rating(userId, movieId, rating))
      //
      (fields(3).toLong % 10, Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble))
    }

    val movies = sc.textFile(movieLensHomeDir + "/movies.dat").map { line =>
      val fields = line.split("::")
      // MovieID::Title::Genres
      // e.g.
      // 1::Toy Story (1995)::Animation|Children's|Comedy
      //  read in movie ids and titles only
      // format: (movieId, movieName)
      (fields(0).toInt, fields(1))
    }.collect.toMap

    val numRatings = ratings.count
    // _._2 is the RDD ratings's Rating in the (Int, Rating) pairs
    // The Rating class is a wrapper around tuple (user: Int, product: Int, rating: Double)
    val numUsers = ratings.map(_._2.user).distinct.count
    val numMovies = ratings.map(_._2.product).distinct.count

    println("\nGot " + numRatings + " ratings from "
      + numUsers + " users on " + numMovies + " movies.")

    // sample a subset of most rated movies for rating elicitation

    // count ratings received for each movie and sort movies by rating counts
    // The Rating class is a wrapper around tuple (user: Int, product: Int, rating: Double)
    // product: Int is the movie id
    val mostRatedMovieIds = ratings.map(_._2.product) // extract movie ids
                                   .countByValue      // count ratings per movie
                                   .toSeq             // convert map to Seq
                                   .sortBy(- _._2)    // sort by rating count
                                   .take(50)          // take 50 most rated
                                   .map(_._1)         // get their ids

    val random = new Random(0)

    val selectedMovies = mostRatedMovieIds.filter(x => random.nextDouble() < 0.2) // randomly select some movies
                                          .map(x => (x, movies(x))) // Get the entire movie info from map movies
                                          .toSeq

    // elicitate ratings

    val myRatings = elicitateRatings(selectedMovies)

    // The ratings are converted to a RDD[Rating] instance via sc.parallelize.
    val myRatingsRDD = sc.parallelize(myRatings, 1)
    // Parallelized collections are created by calling SparkContext's parallelize method on an existing collection
    // The elements of the collection are copied to form a distributed dataset that can be operated on in parallel.
    // Once created, the distributed dataset (myRatingsRDD) can be operated on in parallel.
    // One important parameter for parallel collections is the number of partitions to cut the dataset into.
    // Spark will run one task for each partition of the cluster.
    // you can also set it manually by passing it as a second parameter to parallelize().


    // We will use MLlib’s ALS to train a MatrixFactorizationModel, 
    // which takes a RDD[Rating] object as input. 
    // ALS has training parameters such as rank for matrix factors and regularization constants. 
    // To determine a good combination of the training parameters,
    // we split ratings into train (60%), validation (20%), and test (20%) based on the 
    // last digit of the timestamp, add myRatings to train, and cache them

    val numPartitions = 20
    // ratings format // format: (timestamp % 10, Rating(userId, movieId, rating))
    // The training set is 60%, it is based on the last digit of the timestamp
    // change to 30%, 10% and 10%
    val training = ratings.filter(x => x._1 <= 3)
                          .values
                          .union(myRatingsRDD)
                          .repartition(numPartitions)
                          .persist
    // val validation = ratings.filter(x => x._1 >= 3 && x._1 < 8)
    val validation = ratings.filter(x => x._1 == 4 )
                            .values
                            .repartition(numPartitions)
                            .persist
    // val test = ratings.filter(x => x._1 >= 8).values.persist
    val test = ratings.filter(x => x._1 == 5).values.persist

    val numTraining = training.count
    val numValidation = validation.count
    val numTest = test.count

    println("\nTraining: " + numTraining + " ratings, validation: " + numValidation + " ratings, test: " + numTest + " ratings.")

    // train models and evaluate them on the validation set
    // we will test only 8 combinations resulting from the cross product of 2 different ranks (8 and 12)
    // use rank 12 to reduce the running time
    // val ranks = List(8, 12)
    val ranks = List(12)

    // 2 different lambdas (1.0 and 10.0)
    val lambdas = List(0.1, 10.0)

    // two different numbers of iterations (10 and 20)
    // use numIters 20 to reduce the running time
    //  val numIters = List(10, 20)
    val numIters = List(20)

    // We use the provided method computeRmse to compute the RMSE on the validation set for each model.
    // The model with the smallest RMSE on the validation set becomes the one selected 
    // and its RMSE on the test set is used as the final metric
    // import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
    var bestModel: Option[MatrixFactorizationModel] = None
    var bestValidationRmse = Double.MaxValue
    var bestRank = 0
    var bestLambda = -1.0
    var bestNumIter = -1
    for (rank <- ranks; lambda <- lambdas; numIter <- numIters) {
      // in object ALS
      // def train(ratings: RDD[Rating], rank: Int, iterations: Int, lambda: Double) : MatrixFactorizationModel
      val model = ALS.train(training, rank, numIter, lambda)

      // def computeRmse(model: MatrixFactorizationModel, data: RDD[Rating], n: Long)
      // return  math.sqrt, type is double
      // model is from training.
      val validationRmse = computeRmse(model, validation, numValidation)
      // println("RMSE (validation) = " + validationRmse + " for the model trained with rank = " 
      //   + rank + ", lambda = " + lambda + ", and numIter = " + numIter + ".")
      if (validationRmse < bestValidationRmse) {
        bestModel = Some(model)
        bestValidationRmse = validationRmse
        bestRank = rank
        bestLambda = lambda
        bestNumIter = numIter
      }
    }

    // evaluate the best model on the test set

    // val testRmse = computeRmse(bestModel.get, test, numTest)

    // println("The best model was trained with rank = " + bestRank + " and lambda = " + bestLambda
    //   + ", and numIter = " + bestNumIter + ", and its RMSE on the test set is " + testRmse + ".")

    // create a naive baseline and compare it with the best model

    // val meanRating = training.union(validation).map(_.rating).mean
    // val baselineRmse = math.sqrt(test.map(x => (meanRating - x.rating) * (meanRating - x.rating))
    //                                 .reduce(_ + _) / numTest)
    // val improvement = (baselineRmse - testRmse) / baselineRmse * 100
    // println("The best model improves the baseline by " + "%1.2f".format(improvement) + "%.")

    // make personalized recommendations
    // in class MatrixFactorizationModel
    // def predict(userProducts: RDD[(Int, Int)]): RDD[Rating] 

    val myRatedMovieIds = myRatings.map(_.product).toSet
    val candidates = sc.parallelize(movies.keys.filter(!myRatedMovieIds.contains(_)).toSeq)
    val recommendations = bestModel.get
                                   .predict(candidates.map((0, _)))
                                   .collect
                                   .sortBy(- _.rating)
                                   .take(50)

    var i = 1
    println("\nMovies recommended for you:")
    recommendations.foreach { r =>
      println("\t" + "%2d".format(i) + ": " + movies(r.product))
      i += 1
    }

    println("\n")
    // clean up
    sc.stop();
  }

  /** Compute RMSE (Root Mean Squared Error). */
  def computeRmse(model: MatrixFactorizationModel, data: RDD[Rating], n: Long) = {
    val predictions: RDD[Rating] = model.predict(data.map(x => (x.user, x.product)))
    val predictionsAndRatings = predictions.map(x => ((x.user, x.product), x.rating))
                                           .join(data.map(x => ((x.user, x.product), x.rating)))
                                           .values
    math.sqrt(predictionsAndRatings.map(x => (x._1 - x._2) * (x._1 - x._2)).reduce(_ + _) / n)
  }

  /** Elicitate ratings from command-line. */
  // For each of the selected movies, we will ask you to give a rating (1-5) or 0 if you have never watched this movie. 
  // The method eclicitateRatings returns your ratings, where you receive a special user id 0. 
  //
  def elicitateRatings(movies: Seq[(Int, String)]) = {
    val prompt = "\nPlease rate the following movie (1-5 (best), or 0 if not seen):"
    println(prompt)
    // flatMap, takes every elements to a map
    // ratings format: (timestamp % 10, Rating(userId, movieId, rating))
    val ratings = movies.flatMap { x =>
      var rating: Option[Rating] = None
      var valid = false
      while (!valid) {
        print("\t" + x._2 + ": ")
        try {
          val r = Console.readInt
          if (r < 0 || r > 5) {
            println(prompt)
          } else {
            valid = true
            if (r > 0) {
              // Class Some[A] represents existing values of type A.
              // The Rating class is a wrapper around tuple (user: Int, product: Int, rating: Double)
              // you receive a special user id 0
              rating = Some(Rating(0, x._1, r))
            }
          }
        } catch {
          case e: Exception => println(prompt)
        }
      }
      rating match {
        // An iterator is not a collection, but rather a way to access the elements of a collection one by one.
        case Some(r) => Iterator(r)
        case None => Iterator.empty
      }
    }
    if(ratings.isEmpty) {
      error("No rating provided!")
    } else {
      ratings
    }
  }
}
