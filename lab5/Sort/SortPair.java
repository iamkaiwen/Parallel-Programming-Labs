package page_rank;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.Text;

public class SortPair implements WritableComparable{
	private Text word;
	private double value;

	public SortPair() {
		word = new Text();
		value = 0.0;
	}

	public SortPair(Text word, double value) {
		//TODO: constructor
		this.word = word;
		this.value = value;
	}

	@Override
	public void write(DataOutput out) throws IOException {
		this.word.write(out);
		out.writeDouble(this.value);

	}

	@Override
	public void readFields(DataInput in) throws IOException {
		this.word.readFields(in);
		this.value = in.readDouble();
	}

	public Text getWord() {
		return word;
	}

	public double getValue() {
		return value;
	}

	@Override
	public int compareTo(Object o) {

		double thisValue = this.getValue();
		double thatValue = ((SortPair)o).getValue();

		Text thisWord = this.getWord();
		Text thatWord = ((SortPair)o).getWord();

		// Compare between two objects
		// First order by value, and then sort them lexicographically in ascending order
		if(thisValue == thatValue){
			return String.valueOf(thisWord).compareTo(String.valueOf(thatWord));
		}else if(thisValue > thatValue){
			return -1;
		}else{
			return 1;
		}
	}
} 
