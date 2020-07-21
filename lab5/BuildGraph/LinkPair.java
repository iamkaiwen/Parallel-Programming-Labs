package page_rank;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.Text;

public class LinkPair implements WritableComparable {

	private Text node;
	private int type;

	public LinkPair() {
		
	}
	
	public LinkPair(Text node, int type) {
		//TODO: constructor
		this.node = node;
		this.type = type;
	}

	@Override 
	public void write(DataOutput out) throws IOException {
		this.node.write(out);
		out.writeInt(this.type);
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		this.node = new Text();
		this.node.readFields(in);
		this.type = in.readInt();
	}
	
	public Text getNode() {
		return node;
	}
	
	public int getType() {
		return type;
	}

	@Override
	public int compareTo(Object o) {

		Text thisNode = this.getNode();
		Text thatNode = ((LinkPair)o).getNode();

		int thisType = this.getType();
		int thatType = ((LinkPair)o).getType();

		// Compare between two objects
		// First order by average, and then sort them lexicographically in ascending order
		if(thisType == thatType){
			return String.valueOf(thisNode).compareTo(String.valueOf(thatNode));
		}else if(thisType > thatType){
			return -1;
		}else{
			return 1;
		}
	}
}
