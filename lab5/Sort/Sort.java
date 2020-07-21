package page_rank;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;
import org.apache.hadoop.io.NullWritable;


public class Sort{
	
	public Sort(){

  	}
	
	public void Sort(String[] args , int num_iters) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "Sort");
        job.setJarByClass(Sort.class);

        job.setInputFormatClass(KeyValueTextInputFormat.class);	

        // set the class of each stage in mapreduce
        //job.setMapperClass(xxx.class);
        job.setMapperClass(SortMapper.class);
        //job.setPartitionerClass(xxx.class);
        job.setPartitionerClass(SortPartitioner.class);
        //job.setReducerClass(xxx.class);
        job.setReducerClass(SortReducer.class);
        
        // set the output class of Mapper and Reducer
        job.setMapOutputKeyClass(SortPair.class);
        job.setMapOutputValueClass(NullWritable.class);
        //job.setOutputKeyClass(xxx.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(DoubleWritable.class);
            
        // set the number of reducer
        job.setNumReduceTasks(20);
        
        // add input/output path
        FileInputFormat.addInputPath(job, new Path(args[1] + String.valueOf(num_iters)));
        FileOutputFormat.setOutputPath(job, new Path(args[2]));
        
        job.waitForCompletion(true);
	}
}