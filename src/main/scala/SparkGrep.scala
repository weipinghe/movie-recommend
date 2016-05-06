import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
//
// How to compile
// sbt clean
// sbt compile 
// sbt package

// $ hadoop fs -ls hdfs://localhost/user/cloudera/myfiles
// Found 1 items
// -rw-r--r--   1 cloudera cloudera     305676 2016-05-05 18:22 hdfs://localhost/user/cloudera/myfiles/mywestUsers.txt

// How to run
// /usr/local/spark/bin/spark-submit --executor-memory 512m --master spark://localhost.localdomain:7077 --driver-memory 512m --class  SparkGrep  /var/lib/hadoop-hdfs/testgrep/target/scala-2.10/sparkgrep_2.10-0.1.jar spark://localhost.localdomain:7077  hdfs://localhost/user/cloudera/myfiles/mywestUsers.txt Richard
//

object SparkGrep {
	def main(args: Array[String]) {
		if (args.length < 3) {
			System.err.println("Usage: SparkGrep <host> <input_file> <match_term>")
			System.exit(1)
		}
		val conf = new SparkConf().setAppName("SparkGrep").setMaster(args(0))
		val sc = new SparkContext(conf)
		val inputFile = sc.textFile(args(1), 2).cache()
		val matchTerm : String = args(2)
		val Matches = inputFile.filter(line => line.contains(matchTerm))
		Matches.collect().foreach(println)
		val numMatches = inputFile.filter(line => line.contains(matchTerm)).count()
		println("%s lines in %s contain %s".format(numMatches, args(1), matchTerm))
		System.exit(0)
	}
}
