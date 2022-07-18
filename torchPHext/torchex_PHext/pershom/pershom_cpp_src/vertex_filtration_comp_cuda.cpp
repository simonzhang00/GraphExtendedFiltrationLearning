#include <torch/extension.h>
//#include "ATen/core/function_schema.h"
// #include <ATen/cuda/CUDAApplyUtils.cuh>

// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <thrust/sort.h>
// #include <thrust/unique.h>
// #include <thrust/device_ptr.h>

#include <phat/compute_persistence_pairs.h>
#include <phat/representations/bit_tree_pivot_column.h>
#include <phat/algorithms/twist_reduction.h>
#include <phat/algorithms/standard_reduction.h>
#include <phat/persistence_pairs.h>
#include <vector>
#include <limits>
#include <future>

#include "vertex_filtration_comp_cuda.h"

//link cut tree:
#include <data_structure/link_cut_tree.hpp>
#include <monoids/max_index.hpp>

using namespace torch;
auto EPS= 1e-20;
namespace VertExtendedFiltCompCuda_link_cut_tree{

    class union_find {
    public:
        union_find(int64_t n) {
            this->count = n;
            this->parent.resize(n);
            for (auto i = 0; i < n; i++) {
                this->parent[i] = i;
            }
            this->rank.resize(n, 0);
        }

        int64_t uf_depth(int64_t x) {
            return rank[x];
        }

        int64_t find(int64_t x) {
            int64_t tmp = x;
            while (parent[tmp] != tmp) {
                parent[tmp] = parent[parent[tmp]];
                tmp = parent[tmp];
            }
            return tmp;
        }

        void link(int64_t x, int64_t y) {
            x = find(x);
            y = find(y);
            if (x == y) {
                return;
            }
            if (rank[x] < rank[y]) {
                parent[x] = y;
            } else if (rank[x] > rank[y]) {
                parent[y] = x;
            } else {
                parent[y] = x;
                rank[x] = rank[x] + 1;
            }
            count--;
        }

        int64_t num_connected_component() const {
            return this->count;
        }


    private:
        int64_t count;
        std::vector <int64_t> parent;
        std::vector <int64_t> rank;
    };

    std::vector <Tensor> compute_pd0(const Tensor &vertex_filtration,
                                const std::vector <Tensor> &boundary_info) {
        auto num_nodes = vertex_filtration.size(0);
        union_find uf = union_find(num_nodes);
        Tensor tensor_edges = boundary_info[0];
        Tensor edge_val = std::get<0>(torch::max(vertex_filtration.index({tensor_edges}), 1));
        Tensor sorted_edge_indices = edge_val.argsort(-1, false);
        const Tensor sorted_edges = tensor_edges.index({sorted_edge_indices});
        edge_val = edge_val.index({sorted_edge_indices});
        auto num_edges = sorted_edges.size(0);
        std::vector <Tensor> pd_0;

        for (auto i = 0; i < num_edges; i++) {
            auto e = sorted_edges[i];
            auto e_val = edge_val[i];
            int64_t u = e[0].item<int64_t>();
            int64_t v = e[1].item<int64_t>();
            int64_t root_u = uf.find(u);
            int64_t root_v = uf.find(v);
            if (root_u == root_v) {
                continue;
            }
            int64_t root = root_u;
            int64_t merged = root_v;
            if (vertex_filtration[root].item<double>() > vertex_filtration[merged].item<double>())
                std::swap(root, merged);
            else if (std::abs(vertex_filtration[root].item<double>() - vertex_filtration[merged].item<double>()) <
                     EPS) {
                if (root > merged)
                    std::swap(root, merged);
            }
            auto merged_val = vertex_filtration[merged];
//        std::cout << "M: " << merged_val.item<double>() << " E: " << e_val.item<double>()<< std::endl;
            Tensor pd_pair = torch::stack({merged_val, e_val});
            pd_0.emplace_back(pd_pair);
            uf.link(root, merged);
        }
        return pd_0;

    }

    std::vector <std::vector<Tensor>> extended_filt_persistence_single(const Tensor &vertex_filtration,
                                                             const std::vector <Tensor> &boundary_info) {

        std::vector <std::vector<Tensor>> pd;
        auto num_nodes = vertex_filtration.size(0);
        union_find uf = union_find(num_nodes);
        link_cut_tree<max_index_monoid<double> > lct(num_nodes);
        for (auto i = 0; i < num_nodes; i++) {
            lct.vertex_set(i, {vertex_filtration[i].item<double>(), i});
        }
        std::vector <size_t> pos_edge_index;
        std::vector <Tensor> pd_0_up = compute_pd0(vertex_filtration, boundary_info);
        std::vector <Tensor> pd_0_down, pd_0_ext_plus, pd_1_ext;
        Tensor tensor_edges = boundary_info[0];
        Tensor edge_val = std::get<0>(torch::min(vertex_filtration.index({tensor_edges}), 1));
        Tensor sorted_edge_indices = edge_val.argsort(-1, true);
        const Tensor sorted_edges = tensor_edges.index({sorted_edge_indices});
        edge_val = edge_val.index({sorted_edge_indices});
        auto num_edges = sorted_edges.size(0);
        for (auto i = 0; i < num_edges; i++) {
            auto e = sorted_edges[i];
            auto e_val = edge_val[i];
            int64_t u = e[0].item<int64_t>();
            int64_t v = e[1].item<int64_t>();
            int64_t root_u = uf.find(u);
            int64_t root_v = uf.find(v);
            if (root_u == root_v) {
//            std::cout<<"u and v form positive edge: "<<u<<" "<<v<<std::endl;
                pos_edge_index.push_back(i);
                continue;
            }
            auto u_rank = uf.uf_depth(u);//depth of union find connected component is in the union find data structure
            auto v_rank = uf.uf_depth(v);//depth of union find connected component is in the union find data structure

            if (u_rank < v_rank) {
                lct.evert(u);
                lct.link(u, v);
            } else {
                lct.evert(v);
                lct.link(v, u);
            }

            int64_t root = root_u;
            int64_t merged = root_v;
            if (vertex_filtration[root].item<double>() < vertex_filtration[merged].item<double>())
                std::swap(root, merged);
            else if (std::abs(vertex_filtration[root].item<double>() - vertex_filtration[merged].item<double>()) <
                     EPS) {
                if (root < merged)
                    std::swap(root, merged);
            }
            auto merged_val = vertex_filtration[merged];
            Tensor pd_pair = torch::stack({merged_val, e_val});
            pd_0_down.emplace_back(pd_pair);
            uf.link(root, merged);
        }

        //count the min and max per connected component: this is pd_0_ext_plus
        std::vector <int64_t> connected_components_min(num_nodes, -1);
        std::vector <int64_t> connected_components_max(num_nodes, -1);
        for (auto v = 0; v < num_nodes; v++) {
            auto v_root = uf.find(v);
            if (uf.uf_depth(v_root) > 0) {
                if (connected_components_max[v_root] == -1 && connected_components_min[v_root] == -1) {
                    connected_components_min[v_root] = v;
                    connected_components_max[v_root] = v;
                } else {
                    if (vertex_filtration[v].item<double>() >
                        vertex_filtration[connected_components_max[v_root]].item<double>()) {
                        connected_components_max[v_root] = v;
                    }
                    if (vertex_filtration[v].item<double>() <
                        vertex_filtration[connected_components_min[v_root]].item<double>()) {
                        connected_components_min[v_root] = v;
                    }
                }
            }
        }

        for (auto v = 0; v < num_nodes; v++) {
            auto min_i = connected_components_min[v];
            auto max_i = connected_components_max[v];
            if (min_i != -1 && max_i != -1) {
                pd_0_ext_plus.emplace_back(torch::stack({vertex_filtration[min_i], vertex_filtration[max_i]}));
            }
        }
        if(pos_edge_index.size()>0){
            for (auto ii: pos_edge_index) {
                auto pos_edge = sorted_edges[ii];
                auto pos_edge_val = edge_val[ii];
                int64_t u = pos_edge[0].item<int64_t>();
                int64_t v = pos_edge[1].item<int64_t>();

                //print_graph(lct, num_nodes);
                auto lca = lct.get_lowest_common_ancestor(u, v);
                assert(lca != -1);
                auto p1 = lct.path_get(u, lca);
                auto p2 = lct.path_get(v, lca);

                int64_t critical_vertex;//the maximum on the loop
                int64_t deletion_vertex;//the node to delete whose parent is the critical vertex
                int64_t r;

                if (p1.first > p2.first) {
                    critical_vertex = p1.second;
                    r = lct.get_root(v);
                } else {
                    critical_vertex = p2.second;
                    r = lct.get_root(u);
                }
                deletion_vertex = critical_vertex;

                auto search_node = u;
                bool find_child_of_critical = false;
                while (search_node != critical_vertex && search_node != lca) {
                    auto paren = lct.get_parent(search_node);
                    if (paren == critical_vertex) {
                        deletion_vertex = search_node;
                        break;
                    }
                    search_node = paren;
                }

                if (search_node == lca) {
                    search_node = v;
                    while (search_node != critical_vertex && search_node != lca) {
                        auto paren = lct.get_parent(search_node);
                        if (paren == critical_vertex) {
                            deletion_vertex = search_node;
                            break;
                        }
                        search_node = paren;
                    }
                }

    //        print_graph(lct, num_nodes);
                lct.cut(deletion_vertex);
    //        print_graph(lct, num_nodes);


                auto u_rank = uf.uf_depth(u);//depth of union find connected component is in the union find data structure
                auto v_rank = uf.uf_depth(v);//depth of union find connected component is in the union find data structure

                if (u_rank < v_rank) {
                    lct.evert(u);
                    lct.link(u, v);
                } else {
                    lct.evert(v);
                    lct.link(v, u);
                }

                auto cut_edge_val = vertex_filtration[critical_vertex];
                //std::cout << "CE " << cut_edge_val.item<double>() << " AE " << pos_edge_val.item<double>() << std::endl;
                auto pers_pair = torch::stack({cut_edge_val, pos_edge_val});
                pd_1_ext.push_back(pers_pair);
            }
        }

        if(pd_0_up.size()==0){
            pd_0_up.push_back(torch::tensor({1.0, 1.0}));
        }
        if(pd_0_down.size()==0){
            pd_0_down.push_back(torch::tensor({1.0, 1.0}));
        }
        if(pd_0_ext_plus.size()==0){
            pd_0_ext_plus.push_back(torch::tensor({1.0, 1.0}));
        }
        if(pd_1_ext.size()==0){
            pd_1_ext.push_back(torch::tensor({1.0, 1.0}));
        }
        pd.push_back(pd_0_up);
        pd.push_back(pd_0_down);
        pd.push_back(pd_0_ext_plus);
        pd.push_back(pd_1_ext);
        return pd;

    }

    std::vector <std::vector<std::vector < Tensor>>>
    extended_filt_persistence_batch(const std::vector <std::tuple<Tensor, std::vector < Tensor>>> & batch) {
        auto futures = std::vector < std::future < std::vector < std::vector < Tensor >> >> ();
        for (auto &arg: batch) {
            futures.
            push_back(
            async(std::launch::async, [=] {
                      return extended_filt_persistence_single(
                              std::get<0>(arg),
                              std::get<1>(arg)
                      );
                  }
               )
        );
    }
    auto ret = std::vector < std::vector < std::vector < Tensor>>>();
    for (auto &fut: futures){
        ret.push_back(fut.get());
    }
    return ret;
    }
}