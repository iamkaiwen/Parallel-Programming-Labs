package page_rank;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;


public class Calculate{
	
	public Calculate(){

  	}
	
	public int Calculate(String[] args) throws Exception {
        Configuration conf = new Configuration();
        int p = 0;
        int MAX_ITERS = Integer.parseInt(args[3]);
		for(p = 1;p <= MAX_ITERS || MAX_ITERS == -1;p++){
            Job job = Job.getInstance(conf, "Calculate");
            job.setJarByClass(Calculate.class);

            job.setInputFormatClass(KeyValueTextInputFormat.class);	

            // set the class of each stage in mapreduce
            //job.setMapperClass(xxx.class);
            job.setMapperClass(CalculateMapper.class);
            //job.setPartitionerClass(xxx.class);
            job.setPartitionerClass(CalculatePartitioner.class);
            //job.setReducerClass(xxx.class);
            job.setReducerClass(CalculateReducer.class);
            
            // set the output class of Mapper and Reducer
            job.setMapOutputKeyClass(LinkPair.class);
            job.setMapOutputValueClass(Text.class);
            //job.setOutputKeyClass(xxx.class);
            job.setOutputKeyClass(Text.class);
            job.setOutputValueClass(Text.class);
            
            // set the number of reducer
            job.setNumReduceTasks(27);
            
            // add input/output path
            FileInputFormat.addInputPath(job, new Path(args[1] + String.valueOf(p - 1)));
            FileOutputFormat.setOutputPath(job, new Path(args[1] + String.valueOf(p)));
            
            job.waitForCompletion(true);
            long err = job.getCounters().findCounter(ERROR.ERR).getValue();
            if(MAX_ITERS == -1 && err < 1000000000000L){ // 0.001 * 1000000000000000
                break;
            }
        }
        if(MAX_ITERS == -1){
            return p;
        }else{
            return MAX_ITERS - 1;
        }
	}
}
