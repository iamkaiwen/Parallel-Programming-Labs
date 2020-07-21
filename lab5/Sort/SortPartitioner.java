package page_rank;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.io.NullWritable;

public class SortPartitioner extends Partitioner<SortPair, NullWritable> {
	
	private double maxValue = 100.0;
	private double minValue = 0.0;

	@Override
	public int getPartition(SortPair key, NullWritable value, int numReduceTasks) {
		
		int num = 0;
		
		// customize which <K ,V> will go to which reducer
		// Based on defined min/max value and numReduceTasks
		double interval = (maxValue - minValue) / (double)(numReduceTasks);
		double up = minValue + interval;
		
		for(num = numReduceTasks - 1;num > 0;num--){
			if(key.getValue() < up){
				break;
			}else{
				up += interval;
			}
		}
		
		return num;
	}
}
