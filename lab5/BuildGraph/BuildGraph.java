package page_rank;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;


public class BuildGraph{
	
	public BuildGraph(){

  	}
	
	public void BuildGraph(String[] args) throws Exception {
		Configuration conf = new Configuration();
		
		Job job = Job.getInstance(conf, "BuildGraph");
		job.setJarByClass(BuildGraph.class);

		// set the class of each stage in mapreduce
		//job.setMapperClass(xxx.class);
		job.setMapperClass(GraphMapper.class);
		//job.setPartitionerClass(xxx.class);
		job.setPartitionerClass(GraphPartitioner.class);
		//job.setReducerClass(xxx.class);
		job.setReducerClass(GraphReducer.class);
		
		// set the output class of Mapper and Reducer
		job.setMapOutputKeyClass(LinkPair.class);
		job.setMapOutputValueClass(Text.class);
		//job.setOutputKeyClass(xxx.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		
		// set the number of reducer
		job.setNumReduceTasks(27);
		
		// add input/output path
		FileInputFormat.addInputPath(job, new Path(args[0]));
		// FileOutputFormat.setOutputPath(job, new Path(args[2]));
		FileOutputFormat.setOutputPath(job, new Path(args[1] + String.valueOf(0)));

		job.waitForCompletion(true);
	}
	
}
