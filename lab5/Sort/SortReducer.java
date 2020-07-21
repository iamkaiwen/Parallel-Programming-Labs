package page_rank;

import java.io.IOException;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.io.NullWritable;

public class SortReducer extends Reducer<SortPair, NullWritable, Text, DoubleWritable> {
	private DoubleWritable result = new DoubleWritable();
    public void reduce(SortPair key, Iterable<NullWritable> values, Context context) throws IOException, InterruptedException {
		// output the word and value
		result.set(key.getValue());
		context.write(key.getWord() , result);
	}
}
