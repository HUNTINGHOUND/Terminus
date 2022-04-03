#include "adaboosting.hpp"
#include "decisionStump.hpp"
#include "util.hpp"



double AdaBoosting::compute_error(std::vector<bool> const & incorrect) {
    if(incorrect.size() != weight.size())
        throw std::invalid_argument("Vector size not equal");
    
    double sum = 0;
    
    for(size_t i = 0; i < incorrect.size(); i++) sum += incorrect[i] * weight[i];
    return sum;
}

double AdaBoosting::compute_voting_power(double error) {
    return (1.0 / 2.0) * std::log((1 - error) / error);
}

void AdaBoosting::change_weight(std::vector<bool> const & incorrect, std::vector<double> & w, double a) {
    if(incorrect.size() != w.size())
        throw std::invalid_argument("Vector size not equal");
    
    double incorrect_exp = std::exp(a);
    double correct_exp = std::exp(-a);
        
    double sum = 0;
    for(size_t i = 0; i < w.size(); i++) {
        w[i] *= incorrect[i] ? incorrect_exp : correct_exp;
        sum += w[i];
    }
    
    // normalize weight
    for(size_t i = 0; i < w.size(); i++) {
        w[i] /= sum;
        w[i] *= 100; // make sure that we don't approach precision boundary
    }
}

AdaBoosting::SplitInfo AdaBoosting::find_gini(std::vector<std::vector<double>> const & data,
                                              std::unordered_set<size_t> const & skip, size_t target) {
    SplitInfo min_gini;
    min_gini.gini = INT_MAX;
    for(size_t i = 0; i < data.size(); i++) {
        if(i == target) continue;
        if(skip.count(i)) continue;
        std::vector<double> const & feature = data[i];
        std::vector<double> thresholds = find_threshold(feature); // TODO: Move this to main training loop so that we don't have to recalculate this
        for(double thres : thresholds) {
            std::vector<size_t> d = find_decision(data, i, thres, target);
            double gini = calculate_gini(d);
            if(gini < min_gini.gini) {
                min_gini.feature = i;
                min_gini.threshold = thres;
                min_gini.up = d[0] <= d[1];
                min_gini.deci = std::move(d);
                min_gini.gini = gini;
            }
        }
        
    }
    
    return min_gini;
}

std::vector<double> AdaBoosting::find_threshold(std::vector<double> const & val) {
    std::set<double> sorted(val.begin(), val.end());
    std::vector<double> poss;
    for(auto it = sorted.begin(); it != sorted.end(); it++) {
        if(it != sorted.begin()) {
            poss.push_back(*prev(it) + (*it - *prev(it)) / 2);
        }
    }
    
    return poss;
}

std::vector<size_t> AdaBoosting::find_decision(std::vector<std::vector<double>> const & val, size_t col, double thres, size_t target) {
    std::vector<size_t> res(4, 0);
    for(size_t i : boot) {
        if(val[col][i] >= thres) {
            if(val[target][i] == 0) res[0]++;
            else res[1]++;
        } else {
            if(val[target][i] == 0) res[2]++;
            else res[3]++;
        }
    }
    
    return res;
}

double AdaBoosting::calculate_gini(std::vector<size_t> const & decision) {
    double d_h = (1.0 * decision[1]) / (decision[0] + decision[1] * 1.0);
    double e_h = 1 - d_h;
    double gl_h = 1 - (d_h * d_h) - (e_h * e_h);
    
    double d_l = (1.0 * decision[3]) / (decision[2] + decision[3] * 1.0);
    double e_l = 1 - d_l;
    double gl_l = 1 - (d_l * d_l) - (e_l * e_l);
    
    return (gl_h * (decision[0] + decision[1]) + gl_l * (decision[2] + decision[3])) / (decision[0] + decision[1] + decision[2] + decision[3] * 1.0);
}

std::pair<std::vector<size_t>, std::vector<double>> AdaBoosting::boot_strap(std::vector<std::vector<double>> const & data) {
    std::vector<std::pair<double, size_t>> prefix;
    double sum = 0;
    for(size_t i = 0; i < data[0].size(); i++) {
        prefix.emplace_back(sum, i);
        sum += weight[i];
    }
    
    
    std::vector<size_t> sample;
    std::vector<double> sample_weight;
    std::uniform_real_distribution<double> unif(0, 100);
    std::default_random_engine re;
    for(size_t i = 0; i < data[0].size(); i++) {
        double random_weight = unif(re);
        auto it = std::upper_bound(prefix.begin(), prefix.end(), std::make_pair(random_weight, 0), [](std::pair<double, size_t> const & a, std::pair<double, size_t> const & b) {
            return a.first < b.first;
        });
        
        sample.push_back(std::prev(it)->second);
        sample_weight.push_back(weight[std::prev(it)->second]);
    }
    
    return std::make_pair(std::move(sample), std::move(sample_weight));
}
