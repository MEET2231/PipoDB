#include "vectordb/HNSWIndex.h"
#include "vectordb/Distance.h"
#include "vectordb/Storage.h"
#include <queue>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace vectordb {

    // 1. Initialize hyperparameters and random generator
    HNSWIndex::HNSWIndex(size_t M, size_t ef_construction, size_t ef_search)
        : M_(M), 
          M_max_0_(2 * M), 
          ef_construction_(ef_construction), 
          ef_search_(ef_search),
          max_graph_level_(-1), 
          is_empty_(true),
          rng_(1337), // Fixed seed for reproducibility during testing
          level_dist_(0.0, 1.0) 
    {
        mult_ = 1.0 / std::log(1.0 * M_);
    }

    // 2. Mathematical logic to assign a level to a new vector
    int HNSWIndex::get_random_level() const {
        double r = level_dist_(rng_);
        // Prevent log(0)
        if (r == 0.0) r = 0.000001; 
        
        double f = -std::log(r) * mult_;
        return static_cast< int >(f);
    }

    size_t HNSWIndex::size() const {
        return nodes_.size();
    }
    // 
    // Search for the 'ef' closest neighbors of a query vector starting from a given entry point at a specific layer
    // 
    std::vector< SearchResult > HNSWIndex::search_layer(
        const std::vector< float >& query, 
        VectorID entry_point, 
        int ef, 
        int layer) const 
    {
        std::unordered_set< VectorID > visited;
        visited.insert(entry_point);

        const Node& ep_node = nodes_.at(entry_point);
        float ep_dist = Distance::euclidean(query.data(), ep_node.data.data(), query.size());

        // Min-Heap: Closest nodes we should explore next
        auto min_cmp = [](const SearchResult& a, const SearchResult& b) { return a.distance > b.distance; };
        std::priority_queue< SearchResult, std::vector< SearchResult >, decltype(min_cmp) > candidates(min_cmp);

        // Max-Heap: The 'ef' best nodes we have found so far (worst is at the top to be popped)
        auto max_cmp = [](const SearchResult& a, const SearchResult& b) { return a.distance < b.distance; };
        std::priority_queue< SearchResult, std::vector< SearchResult >, decltype(max_cmp) > top_results(max_cmp);

        candidates.push(SearchResult{ep_dist, entry_point});
        top_results.push(SearchResult{ep_dist, entry_point});

        while (!candidates.empty()) {
            // Extract the closest un-explored candidate
            SearchResult current = candidates.top();
            candidates.pop();

            SearchResult furthest = top_results.top();
            
            // If the closest candidate is further away than our worst result, 
            // going deeper in this direction is useless. We can stop.
            if (current.distance > furthest.distance) {
                break;
            }

            // Get the neighbors of the current node at the specified layer
            const Node& curr_node = nodes_.at(current.id);
            const auto& neighbors = curr_node.neighbors[layer];

            for (VectorID neighbor_id : neighbors) {
                // Only process neighbors we haven't seen yet
                if (visited.find(neighbor_id) == visited.end()) {
                    visited.insert(neighbor_id);

                    const Node& neighbor_node = nodes_.at(neighbor_id);
                    float dist = Distance::euclidean(query.data(), neighbor_node.data.data(), query.size());

                    furthest = top_results.top();
                    
                    // If we have room in our list, OR this neighbor is closer than our worst candidate:
                    if (top_results.size() < static_cast< size_t >(ef) || dist < furthest.distance) {
                        candidates.push(SearchResult{dist, neighbor_id});
                        top_results.push(SearchResult{dist, neighbor_id});

                        // Keep the list size constrained to exactly 'ef'
                        if (top_results.size() > static_cast< size_t >(ef)) {
                            top_results.pop();
                        }
                    }
                }
            }
        }

        // Convert the Max-Heap into a sorted vector (closest first)
        std::vector< SearchResult > results;
        results.reserve(top_results.size());
        while (!top_results.empty()) {
            results.push_back(top_results.top());
            top_results.pop();
        }
        std::reverse(results.begin(), results.end());
        return results;
    }
    // 
    // Search for the 'k' nearest neighbors of a query vector, starting from the entry point and traversing down to layer 0
    // 
    std::vector< SearchResult > HNSWIndex::search(const std::vector< float >& query, int k) const {
        if (is_empty_ || k <= 0) {
            return {};
        }

        VectorID current_ep = entry_point_id_;
        
        // Phase 1: Fast Drop (From max level down to layer 1)
        for (int layer = max_graph_level_; layer > 0; --layer) {
            // ef = 1 because we only want a greedy jump to the single closest node on these highway layers
            std::vector< SearchResult > layer_results = search_layer(query, current_ep, 1, layer);
            current_ep = layer_results[0].id;
        }

        // Phase 2: Detailed Expansion (Layer 0)
        // Ensure our search net (ef) is at least as big as the requested 'k'
        int ef = std::max(static_cast< int >(ef_search_), k);
        std::vector< SearchResult > final_candidates = search_layer(query, current_ep, ef, 0);

        // Trim down to exactly 'k' results if we found more
        if (final_candidates.size() > static_cast< size_t >(k)) {
            final_candidates.resize(k);
        }

        return final_candidates;
    }

    // 
    // Add a new vector to the HNSW graph, wiring it to its neighbors across multiple layers
    // 
    void HNSWIndex::add_vector(const Vector& vec) {
        // Step 1: Initialize the new node
        int target_level = get_random_level();
        
        Node new_node;
        new_node.id = vec.id;
        new_node.data = vec.data;
        // Allocate empty neighbor lists for levels 0 up to target_level
        new_node.neighbors.resize(target_level + 1); 
        
        nodes_[vec.id] = new_node;

        // Step 2: Handle the completely empty graph
        if (is_empty_) {
            entry_point_id_ = vec.id;
            max_graph_level_ = target_level;
            is_empty_ = false;
            return;
        }

        VectorID current_ep = entry_point_id_;
        size_t dim = vec.data.size();
        
        // Step 3: Fast Drop (Navigate without wiring)
        // If the graph is taller than our new node, drop down to target_level
        for (int layer = max_graph_level_; layer > target_level; --layer) {
            std::vector< SearchResult > layer_results = search_layer(vec.data, current_ep, 1, layer);
            current_ep = layer_results[0].id; // Jump to the closest node
        }

        // Step 4: The Wiring Phase
        // Start wiring at the lower of (max_graph_level, target_level)
        int start_layer = std::min(target_level, max_graph_level_);
        
        for (int layer = start_layer; layer >= 0; --layer) {
            // Find a pool of candidates to potentially connect with
            std::vector< SearchResult > candidates = search_layer(vec.data, current_ep, ef_construction_, layer);
            
            // Level 0 gets double the connections (M_max_0) for higher accuracy
            int max_m = (layer == 0) ? static_cast< int >(M_max_0_) : static_cast< int >(M_);
            
            // We only connect up to max_m neighbors
            int num_to_connect = std::min(static_cast< int >(candidates.size()), max_m);
            
            for (int i = 0; i < num_to_connect; ++i) {
                VectorID neighbor_id = candidates[i].id;
                
                // Add forward edge: New Node -> Neighbor
                nodes_[vec.id].neighbors[layer].push_back(neighbor_id);
                
                // Add reverse edge: Neighbor -> New Node
                nodes_[neighbor_id].neighbors[layer].push_back(vec.id);
                
                // PRUNING MATH: Check if we overloaded the neighbor's connections
                if (nodes_[neighbor_id].neighbors[layer].size() > static_cast< size_t >(max_m)) {
                    auto& neighbor_edges = nodes_[neighbor_id].neighbors[layer];
                    std::vector< SearchResult > edge_distances;
                    
                    // Recalculate distance from the neighbor to all of its current connections
                    for (VectorID edge_id : neighbor_edges) {
                        float dist = Distance::euclidean(
                            nodes_[neighbor_id].data.data(), 
                            nodes_[edge_id].data.data(), 
                            dim
                        );
                        edge_distances.push_back({dist, edge_id});
                    }
                    
                    // Sort connections by distance (closest first)
                    std::sort(edge_distances.begin(), edge_distances.end(), 
                              [](const SearchResult& a, const SearchResult& b) { 
                                  return a.distance < b.distance; 
                              });
                    
                    // Keep only the top max_m closest connections; drop the rest
                    neighbor_edges.clear();
                    for (int j = 0; j < max_m; ++j) {
                        neighbor_edges.push_back(edge_distances[j].id);
                    }
                }
            }
            
            // Set the entry point for the next layer down to the closest candidate we just found
            current_ep = candidates[0].id;
        }

        // Step 5: Entry Point Promotion
        // If our new node is taller than the previous highest node, it becomes the new master entry point
        if (target_level > max_graph_level_) {
            entry_point_id_ = vec.id;
            max_graph_level_ = target_level;
        }
    }

    bool HNSWIndex::save(const std::string& filepath) const {
        return Storage::save_hnsw_to_file(filepath, *this);
    }

    bool HNSWIndex::load(const std::string& filepath) {
        return Storage::load_hnsw_from_file(filepath, *this);
    }
}

