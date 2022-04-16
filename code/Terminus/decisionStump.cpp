#include "decisionStump.hpp"

DecisionStump::DecisionStump(std::string const & featureName, size_t feature_idx, double threshold, bool above, bool below) {
    this->feature_idx = feature_idx;
    this->featureName = featureName;
    this->threshold = threshold;
    this->above = above;
    this->below = below;
}

bool DecisionStump::predict(double value) {
    if (value >= threshold) return above;
    return below;
}



