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



public class GraphMapper extends Mapper<LongWritable, Text, LinkPair, Text> {
	
	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
		
		/* unescapeXML */
		String input = this.unescapeXML(String.valueOf(value));
		Text src;
         	
		/*  Match title pattern */  
		Pattern titlePattern = Pattern.compile("<title>(.+?)</title>");
		Matcher titleMatcher = titlePattern.matcher(input);
		// No need capitalizeFirstLetter
		if(titleMatcher.find()){
			src = new Text(titleMatcher.group(1));
		}else{
			return;
		}

		for(int i = 0;i < 27;i++){
			LinkPair lp0 = new LinkPair(src , i);
			context.write(lp0 , new Text());
		}
		
		/*  Match link pattern */
        Pattern linkPattern = Pattern.compile("\\[\\[(.+?)([\\|#]|\\]\\])");
		Matcher linkMatcher = linkPattern.matcher(input);
		LinkPair lp1 = new LinkPair(src , -1);
		context.write(lp1 , new Text());
		// Need capitalizeFirstLetter
		while(linkMatcher.find()){
			String dest = this.capitalizeFirstLetter(linkMatcher.group(1));
			context.write(lp1 , new Text(dest));
		}
	}
	
	private String unescapeXML(String input) {

		return input.replaceAll("&lt;", "<").replaceAll("&gt;", ">").replaceAll("&amp;", "&").replaceAll("&quot;", "\"").replaceAll("&apos;", "\'");

    }

    private String capitalizeFirstLetter(String input){

    	char firstChar = input.charAt(0);

        if ( firstChar >= 'a' && firstChar <='z'){
            if ( input.length() == 1 ){
                return input.toUpperCase();
            }
            else
                return input.substring(0, 1).toUpperCase() + input.substring(1);
        }
        else 
        	return input;
    }
}
