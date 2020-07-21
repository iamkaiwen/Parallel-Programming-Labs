package page_rank;

import java.io.IOException;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import java.util.HashSet;
import java.lang.StringBuffer;


public class GraphReducer extends Reducer<LinkPair,Text,Text,Text> {
	private HashSet <Text> HS = new HashSet<Text>();
	private int num_dangling_node = 0;
    public void reduce(LinkPair key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
        if(key.getType() != -1){
			HS.add(key.getNode());
        }else{
			Text src = key.getNode();

			StringBuffer buf0 = new StringBuffer();
			int count = 0;
			for (Text val: values) {				
				if(HS.contains(val)){
					buf0.append(String.valueOf("|"));
					buf0.append(String.valueOf(val));
					count += 1;
				}
				
			}
			
			if(count == 0){
				num_dangling_node += 1;
			}

			StringBuffer buf1 = new StringBuffer();
			buf1.append((double) 1 / (double) HS.size());
			buf1.append(String.valueOf("|"));
			buf1.append(count);
			buf1.append(String.valueOf("|"));
			buf1.append(HS.size());

			context.write(src , new Text(buf1.toString() + buf0.toString()));
        }
	}
	public void cleanup(Context context) throws IOException, InterruptedException {
		// context.write(new Text("num_dangling_node : ") , new Text(String.valueOf(num_dangling_node)));
	}
}
