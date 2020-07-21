package page_rank;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class PageRank{
	
		public static void main(String[] args) throws Exception {  
		
			/* Don't need to modify this file.
			InputDir : args[0]
			TmpDir : args[1]
			OutputDir : args[2] 
			Number of reducer for Sort : args[3] */

			//Job 1: Build Graph
			BuildGraph job1 = new BuildGraph();
			job1.BuildGraph(args);

			//Job 2: Calculate
			Calculate job2 = new Calculate();
			int num_iters = job2.Calculate(args);

			// Job 3: Sort
			Sort job3 = new Sort();
			job3.Sort(args , num_iters);
			
			System.exit(0);
		}  
}
