//#define PROBLEM "https://judge.yosupo.jp/problem/dynamic_tree_vertex_add_path_sum"
#include "../data_structure/link_cut_tree.hpp"
#include "../monoids/plus.hpp"
#include "../utils/macros.hpp"
#include "../hack/fastio.hpp"
#include <stack>
#include <vector>
#include <iostream>
using namespace std;
void print_graph(link_cut_tree<plus_monoid<int64_t> > lct, int n){
    for(int i=0; i<n ; i++){
        std::cout<<i <<" has parent: " <<lct.get_parent(i)<<std::endl;
        std::cout<<i << " has value: "<<lct.vertex_get(i)<<std::endl;
    }
    std::cout<<"root of n-1 is: "<<lct.get_root(n-1)<<std::endl;
    std::cout<<"LCA of 1 and n-1 is: "<< lct.get_lowest_common_ancestor(1,n-1)<<std::endl;
}
int main() {
    int n= 5;
    link_cut_tree<plus_monoid<int64_t> > lct(n);
    lct.vertex_set(1,1);
    lct.vertex_set(0,0);
    lct.vertex_set(2,2);
    lct.vertex_set(3,3);
    lct.vertex_set(4,4);
    lct.link(1,0);
    lct.link(2,0);
    lct.link(3,2);
    lct.link(4,2);
    print_graph(lct, n);

    lct.cut(2);
    lct.link(2,1);
    print_graph(lct, n);

    //std::cout<<lct.to_graphviz();

    return 0;
}

