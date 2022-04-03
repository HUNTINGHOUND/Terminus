#ifndef adaboosting_hpp
#define adaboosting_hpp

#include "common.hpp"
#include "decisionStump.hpp"


class AdaBoosting {
    
public:
    
    struct SplitInfo {
        size_t feature;
        double threshold;
        bool up;
        
        std::vector<size_t> deci;
        double gini;
    };
    
    
    // Forest of decision stumps
    std::vector<DecisionStump> forrest;
    
    // Voting power of each stump
    std::vector<double> voting_power;
    
    // Weight of each row of data
    std::vector<double> weight;
    
    // Current bootstrap
    std::vector<size_t> boot;
    
    // Find the feature and threshold that yields the least gini index. Return the information.
    SplitInfo find_gini(std::vector<std::vector<double>> const & data, std::unordered_set<size_t> const & skip, size_t target=0);
    
    // Find the thresholds given values of a feature.
    static std::vector<double> find_threshold(std::vector<double> const & val);
    
    // Find the labels of the data given a threshold and a feature.
    // Return a vector of the following form
    // {number of 0 with feature higher than the threshold, number of 1 with feature higher than the threshold,
    //  number of 0 with feature lower than the threshold, number of 1 with feature lower than the threshold }
    std::vector<size_t> find_decision(std::vector<std::vector<double>> const & val, size_t col, double thres, size_t target);
    
    // Calculate the gini index given the purity of the 
    static double calculate_gini(std::vector<size_t> const & decision);
    
    // Compute the error given the incorrect cases
    double compute_error(std::vector<bool> const & incorrect);
    
    // Compute the voting power given the error
    double compute_voting_power(double error);
    
    // Modify weight given the incorrect cases and voting power
    static void change_weight(std::vector<bool> const & incorrect, std::vector<double> & w, double a);
    
    // Bootstrap data, return weighted selection
    std::pair<std::vector<size_t>, std::vector<double>> boot_strap(std::vector<std::vector<double>> const & data);
    
    // TODO: Implement training loop
};

#endif /* adaboosting_hpp */
