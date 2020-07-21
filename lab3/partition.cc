#define MAX_NUM_PROCS 4
#define MAX_NUM_VERTICES 2000
#include <iostream>
#include <cassert>
#include <fstream>
#include <utility>
#include <vector>
#include <map>
#include <algorithm>
int num_procs , num_vertices , num_edges;
int group[MAX_NUM_PROCS] = {0};
std::vector < std::pair<int , int> > edges_vec;
std::map<int , int> index_map;
int get_group(){
    int p = 0 , mini_val = group[0]; 
    for(int i = 1;i < num_procs;i++){
        if(group[i] < mini_val){
            p = i , mini_val = group[i];
        }
    }
    return p;
}
int get_group_with_vertex(int x){
    return (group[index_map[x]] < (int)(num_edges / num_procs))? index_map[x] : get_group();
}
int main(int argc, char** argv){

    assert(argc == 4);
    const char* in_filename = argv[1];
    const char* out_filename = argv[2];
    num_procs = strtol(argv[3], 0, 10);

    std::ifstream in_fs (in_filename, std::ifstream::binary);
    std::ofstream out_fs (out_filename);
    
    in_fs.read((char*)&num_vertices , sizeof(int));
    in_fs.read((char*)&num_edges , sizeof(int));

    for(int i = 0;i < num_edges;i++){
        int src , dest , weight;
        in_fs.read((char*)&src , sizeof(int));
        in_fs.read((char*)&dest , sizeof(int));
        in_fs.read((char*)&weight , sizeof(int));
        edges_vec.push_back(std::make_pair(src , dest));
    }

    std::sort(edges_vec.begin() , edges_vec.end());
    
    std::map<int , int>::iterator src_it , dest_it;
    for(auto x : edges_vec){
        int src = x.first , dest = x.second;
        src_it = index_map.find(src);
        dest_it = index_map.find(dest);
        if(src_it != index_map.end() && dest_it != index_map.end()){
            continue;
        }else if(src_it != index_map.end()){
            index_map[dest] = get_group_with_vertex(src);
        }else if(dest_it != index_map.end()){
            index_map[src] = get_group_with_vertex(dest);
        }else{
            index_map[src] = index_map[dest] = get_group();
        }
        group[index_map[src]]++;
    }
    char c = '\n';
    for(auto x : index_map){
    	out_fs << "0\n";
    }

    in_fs.close();
    out_fs.close();
    return 0;
}

