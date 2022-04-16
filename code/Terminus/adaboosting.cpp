#include "adaboosting.hpp"
#include "decisionStump.hpp"
#include "util.hpp"



double BinaryAdaBoosting::compute_error(std::vector<bool> const & incorrect) {
    if(incorrect.size() != weight.size())
        throw std::invalid_argument("Vector size not equal");
    
    double sum = 0;
    for(size_t i = 0; i < incorrect.size(); i++) sum += incorrect[i] * weight[i];
    return sum;
}

double BinaryAdaBoosting::compute_voting_power(double error) {
    return (1.0 / 2.0) * std::log((100 - error) / error);
}

void BinaryAdaBoosting::change_weight(std::vector<bool> const & incorrect, std::vector<double> & w, double a) {
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

BinaryAdaBoosting::SplitInfo BinaryAdaBoosting::find_gini(std::vector<std::vector<double>> const & data, std::vector<bool> const & target, std::unordered_map<size_t, double> const & skip) {
    SplitInfo min_gini;
    min_gini.gini = INT_MAX;
    for(size_t i = 0; i < data.size(); i++) {
        std::vector<double> const & feature = data[i];
        std::vector<double> thresholds = find_threshold(feature); // TODO: Move this to main training loop so that we don't have to recalculate this
        for(double thres : thresholds) {
            if(skip.count(i) && skip.at(i) == thres) continue;
            
            std::vector<size_t> d = find_decision(data, i, thres, target);
            double gini = calculate_gini(d);
            if(gini < min_gini.gini) {
                min_gini.initialized = true;
                min_gini.feature = i;
                min_gini.threshold = thres;
                min_gini.up = d[0] < d[1];
                min_gini.down = d[2] < d[3];
                min_gini.deci = std::move(d);
                min_gini.gini = gini;
            }
        }
        
    }
    
    return min_gini;
}

std::vector<double> BinaryAdaBoosting::find_threshold(std::vector<double> const & val) {
    std::set<double> sorted(val.begin(), val.end());
    std::vector<double> poss;
    for(auto it = sorted.begin(); it != sorted.end(); it++) {
        if(it != sorted.begin()) {
            poss.push_back(*prev(it) + (*it - *prev(it)) / 2);
        }
    }
    
    return poss;
}

std::vector<size_t> BinaryAdaBoosting::find_decision(std::vector<std::vector<double>> const & val, size_t col, double thres, std::vector<bool> const & target) {
    std::vector<size_t> res(4, 0);
    for(size_t i : boot) {
        if(val[col][i] >= thres) {
            if(target[i] == 0) res[0]++;
            else res[1]++;
        } else {
            if(target[i] == 0) res[2]++;
            else res[3]++;
        }
    }
    
    return res;
}

double BinaryAdaBoosting::calculate_gini(std::vector<size_t> const & decision) {
    double d_h = (1.0 * decision[1]) / (decision[0] + decision[1] * 1.0);
    double e_h = 1 - d_h;
    double gl_h = 1 - (d_h * d_h) - (e_h * e_h);
    
    double d_l = (1.0 * decision[3]) / (decision[2] + decision[3] * 1.0);
    double e_l = 1 - d_l;
    double gl_l = 1 - (d_l * d_l) - (e_l * e_l);
    
    return (gl_h * (decision[0] + decision[1]) + gl_l * (decision[2] + decision[3])) / (decision[0] + decision[1] + decision[2] + decision[3] * 1.0);
}

std::vector<size_t> BinaryAdaBoosting::boot_strap(size_t n) {
    std::vector<std::pair<double, size_t>> prefix;
    double sum = 0;
    for(size_t i = 0; i < n; i++) {
        prefix.emplace_back(sum, i);
        sum += weight[i];
    }
    prefix.emplace_back(sum, n);
    
    
    std::vector<size_t> sample;
    std::uniform_real_distribution<double> unif(0, 100);
    std::default_random_engine re;
    for(size_t i = 0; i < n; i++) {
        double random_weight = unif(re);
        auto it = std::upper_bound(prefix.begin(), prefix.end(), std::make_pair(random_weight, 0), [](std::pair<double, size_t> const & a, std::pair<double, size_t> const & b) {
            return a.first < b.first;
        });
        
        sample.push_back(it->second - 1);
    }
    
    return sample;
}

void BinaryAdaBoosting::fit(std::vector<std::vector<double>> const & x, std::vector<bool> const & y, size_t max_step, std::vector<std::string> const & feature_name) {
    // clean up previous training
    forrest.clear();
    voting_power.clear();
    
    //proper training
    size_t n = x[0].size();
    boot = std::vector<size_t>(n);
    std::iota(boot.begin(), boot.end(), 0);
    weight = std::vector<double>(n, 1.0 / x[0].size() * 100);
    
    std::unordered_map<size_t, double> seen_thres;
    size_t i = 0;
    while(i < max_step) {
        
        SplitInfo thres_info = find_gini(x, y, seen_thres);
        
        // We used all available stump
        if(!thres_info.initialized) break;
        
        forrest.emplace_back(feature_name.size() ? feature_name[thres_info.feature] : ("Feature " + std::to_string(i)), thres_info.feature, thres_info.threshold, thres_info.up, thres_info.down);
        
        std::vector<bool> incorrect;
        for(size_t idx : boot) {
            if((!forrest.back().predict(x[thres_info.feature][idx]) && y[idx]) || (forrest.back().predict(x[thres_info.feature][idx]) && !y[idx]))
                incorrect.push_back(true);
            else incorrect.push_back(false);
        }
        double error = compute_error(incorrect);
        if(!error) {
            voting_power.push_back(1.0);
            break;
        }
        double a = compute_voting_power(error);
        
        voting_power.push_back(a);
        
        change_weight(incorrect, weight, a);
        
        // seen_thres[thres_info.feature] = thres_info.threshold;
        std::vector<size_t> bb = boot_strap(n), new_boot;
        for(size_t idx : bb) new_boot.push_back(boot[idx]);
        std::swap(new_boot, boot);
        i++;
    }
}

std::vector<bool> BinaryAdaBoosting::predict(std::vector<std::vector<double>> const & x) {
    std::vector<bool> y_hat;
    for(size_t i = 0; i < x[0].size(); i++) {
        double tsum = 0, fsum = 0;
        for(size_t j = 0; j < forrest.size(); j++) {
            if(forrest[j].predict(x[forrest[j].feature_idx][i])) {
                tsum += voting_power[j];
            } else {
                fsum += voting_power[j];
            }
        }
        
        double s = tsum + fsum;
        y_hat.push_back(tsum / s >= fsum / s);
    }
    return y_hat;
}
