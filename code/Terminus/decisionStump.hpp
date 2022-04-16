#ifndef decisionStump_hpp
#define decisionStump_hpp

#include "common.hpp"

class DecisionStump {
public:
    
    double threshold;
    std::string featureName;
    size_t feature_idx;
    bool above, below;
    
    
    DecisionStump(std::string const & featureName, size_t feature_idx, double threshold, bool above=true, bool below=false);
    
    bool predict(double value);
};

#endif /* decisionStump_hpp */
