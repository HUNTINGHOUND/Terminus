#ifndef decisionStump_hpp
#define decisionStump_hpp

#include "common.hpp"

class DecisionStump {
    
    double threshold;
    std::string featureName;
    bool above;
    
public:
    
    DecisionStump(std::string const & featureName, double threshold, bool above=true);
    
    bool predict(double value);
};

#endif /* decisionStump_hpp */
