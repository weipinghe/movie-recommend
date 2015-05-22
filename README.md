# movie-recommend
# to run
sbt clean
sbt compile
sbt package

# /var/lib/hadoop-hdfs/movie-recommend/target/scala-2.10/movielens-als_2.10-0.1.jar

/usr/local/spark/bin/spark-submit --executor-memory 512m --master spark://localhost.localdomain:7077 --driver-memory 512m --class MovieLensALS /var/lib/hadoop-hdfs/movie-recommend/target/scala-2.10/movielens-als_2.10-0.1.jar /user/hdfs
