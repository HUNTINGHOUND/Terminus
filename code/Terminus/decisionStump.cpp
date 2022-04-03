#include "decisionStump.hpp"

DecisionStump::DecisionStump(std::string const & featureName, double threshold, bool above) {
    this->featureName = featureName;
    this->threshold = threshold;
    this->above = above;
}

bool DecisionStump::predict(double value) {
    if (value >= threshold) return above;
    return !above;
}



