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
    return (1.0 / 2.0) * std::log((1000 - error) / error);
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
        w[i] *= 1000; // make sure that we don't approach precision boundary
    }
}

BinaryAdaBoosting::SplitInfo BinaryAdaBoosting::find_best_split(std::vector<std::vector<double>> const & data, std::vector<bool> const & target) {
    SplitInfo min_gini;
    min_gini.gini = INT_MAX;
    for(size_t i = 0; i < data.size(); i++) {
        std::vector<double> const & feature = data[i];
        std::vector<double> thresholds = find_threshold(feature);
        for(double thres : thresholds) {
            std::vector<size_t> d = find_decision(data, i, thres, target);
            double gini = calculate_gini(d);
            if(gini < min_gini.gini) {
                min_gini.feature = i;
                min_gini.threshold = thres;
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

void decision_thread(std::atomic_size_t& zero, std::atomic_size_t& one, std::atomic_size_t& two, std::atomic_size_t& three,
                     size_t start, size_t end, size_t col, double thres,
                     std::vector<std::vector<double>> const & val, std::vector<bool> const & target, std::vector<size_t> & boot) {
    for(size_t i = start; i < end; i++) {
        size_t j = boot[i];
        if(val[col][j] >= thres) {
            if(target[j] == 0) zero++;
            else one++;
        } else {
            if(target[j] == 0) two++;
            else three++;
        }
    }
    
}

std::vector<size_t> BinaryAdaBoosting::find_decision(std::vector<std::vector<double>> const & val, size_t col, double thres, std::vector<bool> const & target) {
    std::vector<size_t> res(4, 0);
    std::vector<std::thread> list;
    list.reserve(16);
    std::atomic_size_t zero = 0, one = 0, two = 0, three = 0;
    size_t i = 0;
    size_t amount = boot.size() / 14;
    while(i < boot.size()) {
        list.emplace_back(decision_thread,
                          std::ref(zero), std::ref(one), std::ref(two), std::ref(three),
                          i, std::min(i + amount, boot.size()), col, thres,
                          std::ref(val), std::ref(target), std::ref(boot));
        i = std::min(i + amount, boot.size());
    }
    
    for(size_t t = 0; t < list.size(); t++) {
        list[t].join();
    }
    
    res[0] = zero;
    res[1] = one;
    res[2] = two;
    res[3] = three;
    
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


void boot_strap_thread(size_t n, std::vector<std::pair<double, size_t>> const & prefix, std::vector<size_t>& sample, std::atomic_size_t& idx) {
    std::uniform_real_distribution<double> unif(0, 1000);
    std::default_random_engine re;
    for(size_t i = 0; i < n; i++) {
        double random_weight = unif(re);
        auto it = std::upper_bound(prefix.begin(), prefix.end(), std::make_pair(random_weight, 0), [](std::pair<double, size_t> const & a, std::pair<double, size_t> const & b) {
            return a.first < b.first;
        });
        
        sample[idx.fetch_add(1)] = it->second - 1;
    }
}

std::vector<size_t> BinaryAdaBoosting::boot_strap(size_t n) {
    std::vector<std::pair<double, size_t>> prefix;
    double sum = 0;
    for(size_t i = 0; i < n; i++) {
        prefix.emplace_back(sum, i);
        sum += weight[i];
    }
    prefix.emplace_back(sum, n);
    
    
    std::vector<size_t> sample(n, -1);
    std::vector<std::thread> thread_list;
    thread_list.reserve(16);
    std::atomic_size_t idx = 0;
    size_t i = 0;
    size_t amount = n / 14;
    while(i < n) {
        thread_list.emplace_back(boot_strap_thread, std::min(amount, n - i), std::ref(prefix), std::ref(sample), std::ref(idx));
        i = std::min(n, i + amount);
    }
    
    
    for(size_t t = 0; t < thread_list.size(); t++) thread_list[t].join();
    for(size_t x = 0; x < sample.size(); x++)
        if(sample[x] >= n) throw std::logic_error("bad");
    
    return sample;
}

void BinaryAdaBoosting::fit(std::vector<std::vector<double>> const & x, std::vector<bool> const & y, size_t max_step, std::vector<std::string> const & feature_name, double learning_rate) {
    // clean up previous training
    forrest.clear();
    voting_power.clear();
    
    //proper training
    size_t n = x[0].size();
    boot = std::vector<size_t>(n);
    std::iota(boot.begin(), boot.end(), 0);
    weight = std::vector<double>(n, 1.0 / n * 1000);
    
    size_t i = 0;
    while(i < max_step) {
        
        SplitInfo thres_info = find_best_split(x, y);
        
        forrest.emplace_back(feature_name[thres_info.feature], thres_info.feature, thres_info.threshold);
        
        std::vector<bool> incorrect;
        for(size_t idx : boot) {
            if((!forrest.back().predict(x[thres_info.feature][idx]) && y[idx]) || (forrest.back().predict(x[thres_info.feature][idx]) && !y[idx]))
                incorrect.push_back(true);
            else incorrect.push_back(false);
        }
        double error = compute_error(incorrect);
        if(error <= 0 || error >= 1000) {
            std::cout << "error boundary reached, stopping early" << std::endl;
            forrest.pop_back();
            //voting_power.push_back(*std::max_element(voting_power.begin(), voting_power.end()));
            break;
        }

        double a = compute_voting_power(error) * learning_rate;
        voting_power.push_back(a);
        
        change_weight(incorrect, weight, a);
        
        std::vector<size_t> bb = boot_strap(n), new_boot;
        for(size_t idx : bb)
            new_boot.push_back(boot[idx]);
        std::swap(new_boot, boot);
        phmap::flat_hash_set<size_t> boot_uniq(boot.begin(), boot.end());
        
        if(boot_uniq.size() < 3) {
            std::cout << "Not enough diversity, terminate early" << std::endl;
            break;
        }
         
        std::cout << "Finished stump " << i + 1 << " used feature " << feature_name[thres_info.feature] << " with threshold = " << thres_info.threshold << std::endl;
        i++;
    }
}

std::vector<bool> BinaryAdaBoosting::predict(std::vector<std::vector<double>> const & x, phmap::flat_hash_map<std::string, size_t> & feature_name_to_idx) {
    std::vector<bool> y_hat;
    for(size_t i = 0; i < x[0].size(); i++) {
        if(x[feature_name_to_idx["$"]][i] || x[feature_name_to_idx["%"]][i] ||
           x[feature_name_to_idx[","]][i] || x[feature_name_to_idx["open quote"]][i] ||
           (x[feature_name_to_idx["."]][i] && x[feature_name_to_idx["next is close quote"]][i]) ||
           (x[feature_name_to_idx[":"]][i] && x[feature_name_to_idx["next is quote"]][i]) ||
           (x[feature_name_to_idx["close quote"]][i] && x[feature_name_to_idx["next ("]][i])) {
            y_hat.push_back(false);
            continue;
        }
        if((x[feature_name_to_idx["."]][i] && x[feature_name_to_idx["next is open quote"]][i]) ||
           (x[feature_name_to_idx["''"]][i] && x[feature_name_to_idx["prev ."]][i])) {
            y_hat.push_back(true);
            continue;
        }
        
        double sum = 0;
        for(size_t j = 0; j < forrest.size(); j++) {
            if(forrest[j].predict(x[feature_name_to_idx[forrest[j].featureName]][i])) {
                sum += voting_power[j];
            } else {
                sum -= voting_power[j];
            }
        }
        
        y_hat.push_back(sum >= 0);
    }
    return y_hat;
}
