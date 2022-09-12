#pragma once


#include <torch/extension.h>


using namespace torch;

namespace VertExtendedFiltCompCuda_link_cut_tree{
    std::vector <std::vector<std::vector < Tensor>>>
    extended_filt_persistence_batch(const std::vector <std::tuple<Tensor, std::vector < Tensor>>>&  batch);
}
namespace VertExtendedFiltCompCuda_link_cut_tree_cyclereps{
   std::vector<std::vector <std::vector<std::vector < Tensor>>>>
    extended_filt_persistence_batch(const std::vector <std::tuple<Tensor, std::vector < Tensor>>>&  batch);
}