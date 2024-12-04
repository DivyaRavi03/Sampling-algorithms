CIS - 579 
PROGRAMMING ASSIGNMENT 2

[Divya T R - 60482748]

The input line that needs to be given for the python file is:

python pa2.py --sample-size “samplesize” --runs "runsize (10)"

where --> "pa2.py"        is the name of the python file.

      --> "--sample-size" is the number of samples for all sampling algorithm. By giving sample size [10, 50, 100, 200, 500, 1000, 10000] will run "runs" 			  
			  times for given sample size and return the probability.

      --> "--runs"        is the number of times each sampling algorithm should run. By giving input as 10 will run the sampling algorithm 10 times and 			  
			  return the average of the probabilities.  

The output will look like:

Query 1: Alarm is false, infer Burglary and JohnCalls being true
Exact Inference: ['<B, J, 0.0000500000>', '<B, J, 0.0009500000>', '<B, J, 0.0499500000>', '<B, J, 0.9490500000>']
Results for sampling method: Prior Sampling
Number of samples: 1000, Results: <B, J, 0.0009126820>, <B, J, 0.0483085169>
Results for sampling method: Rejection Sampling
Number of samples: 1000, Results: <B, J, 0.0007069744>, <B, J, 0.0531617061>
Results for sampling method: Likelihood Weighting
Number of samples: 1000, Results: <B, J, 0.0007071140>, <B, J, 0.0505123261>


