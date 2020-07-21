package page_rank;

import java.util.StringTokenizer;
import java.io.IOException;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import java.util.HashSet;
import java.lang.StringBuffer;

enum ERROR {
	ERR
}

public class CalculateReducer extends Reducer<LinkPair,Text,Text,Text> {
    private double dangling_pr = 0.0;
    private double err = 0.0;
    public void reduce(LinkPair key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
        if(key.getType() != -1){
			for (Text val: values) {
                dangling_pr += Double.parseDouble(String.valueOf(val));
            }
        }else{
            Text src = key.getNode();

            double PR = 0, newPR = 0;
            int num_links = 1 , num_nodes = 1;
            StringBuffer buf0 = new StringBuffer();

            for (Text val: values) {
                StringTokenizer itr = new StringTokenizer(String.valueOf(val) , "|");
                if(itr.countTokens() == 1){
                    newPR += Double.parseDouble(itr.nextToken());
                }else{
                    PR = Double.parseDouble(itr.nextToken());
                    num_links = Integer.parseInt(itr.nextToken());
                    num_nodes = Integer.parseInt(itr.nextToken());
                    while(itr.hasMoreTokens()){
                        buf0.append(String.valueOf("|"));
					    buf0.append(String.valueOf(itr.nextToken()));
                    }
                }
            }
            
            newPR *= 0.85;
            newPR += (0.15 + 0.85 * dangling_pr) / (double) num_nodes;
            err += Math.abs(newPR - PR);

			StringBuffer buf1 = new StringBuffer();
			buf1.append(newPR);
			buf1.append(String.valueOf("|"));
			buf1.append(num_links);
			buf1.append(String.valueOf("|"));
			buf1.append(num_nodes);

			context.write(src , new Text(buf1.toString() + buf0.toString()));
        }
	}
	public void cleanup(Context context) throws IOException, InterruptedException {
		context.getCounter(ERROR.ERR).increment((long)(err * 1000000000000000L));
	}
}
