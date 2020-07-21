package page_rank;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Partitioner;

public class CalculatePartitioner extends Partitioner<LinkPair,Text> {
    @Override
    public int getPartition(LinkPair key, Text value, int numReduceTasks) {
        // customize which <K ,V> will go to which reducer
        int type = key.getType();
        if(type == -1){
            int n = String.valueOf(key.getNode()).charAt(0) - 'A';
            if(n >= 0 && n < 26){
                return n;
            }else{
                return 26;
            }
        }else{
            return type;
        }
	}
}
