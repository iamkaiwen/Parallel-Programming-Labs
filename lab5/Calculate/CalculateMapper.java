package page_rank;
import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;

import java.util.Scanner;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.conf.Configuration;

import java.util.ArrayList;
import java.util.Arrays;
import java.net.URI; 
import java.io.*;



public class CalculateMapper extends Mapper<Text, Text, LinkPair, Text> {
    private double dangling_pr = 0.0;
    
	public void map(Text key, Text value, Context context) throws IOException, InterruptedException {
		StringTokenizer itr = new StringTokenizer(String.valueOf(value) , "|");
        double PR = 0; int num_links = 1 , num_nodes = 1;
        PR = Double.parseDouble(itr.nextToken());
        num_links = Integer.parseInt(itr.nextToken());
        num_nodes = Integer.parseInt(itr.nextToken());
        
        if(num_links == 0){
            dangling_pr += PR;
        }else{
            double PR_C = PR / num_links;
            while(itr.hasMoreTokens()){
                Text dest = new Text(itr.nextToken());
                LinkPair lp0 = new LinkPair(dest , -1);
                context.write(lp0 , new Text(String.valueOf(PR_C)));
            }
        }
        LinkPair lp1 = new LinkPair(key , -1);
        context.write(lp1 , value);
    }

    public void cleanup(Context context) throws IOException, InterruptedException {
        for(int i = 0;i < 27;i++){
            LinkPair lp0 = new LinkPair(new Text("dangling_pr") , i);
            context.write(lp0 , new Text(String.valueOf(dangling_pr)));
        }
	}
}
