#include "feature_decoder.hpp"
#include "util.hpp"

std::string FeatureDecoder::raw_read(std::string file_path) {
    std::fstream file(file_path);
    if(!file.is_open()) throw std::invalid_argument("File: " + file_path + " failed to open");
    std::stringstream ss;
    ss << file.rdbuf();
    file.close();
    return ss.str();
}

std::vector<Sentence> WSJDecoder::read_file(std::string file_path) {
    std::fstream file(file_path);
    if(!file.is_open()) throw std::invalid_argument("File: " + file_path + " failed to open");
    
    std::vector<Sentence> res = {Sentence()};
    std::string line;
    while(std::getline(file, line)) {
        if(!line.size()) {
            res.push_back(Sentence());
        } else {
            size_t find_tab = line.find("\t");
            res.back().words.emplace_back(line.substr(0, find_tab), line.substr(find_tab + 1));
        }
    }
    file.close();
    
    return res;
}

std::tuple<std::vector<std::vector<double>>, std::vector<bool>, std::vector<std::string>> WSJDecoder::feature_decode(std::vector<Sentence>& sentences) {
    std::vector<bool> res;
    
    word_end_prob.clear();
    double sym_word_count = 0;
    
    for(auto curr : sentences) {
        size_t i = 0;
        for(auto p : curr.words) {
            if(is_symbol(p.first)) {
                res.push_back(i == curr.words.size() - 1);
                if(i) {
                    word_end_prob[curr.words[i - 1].first] += i == curr.words.size() - 1;
                    sym_word_count++;
                }
            }
            i++;
        }
    }
    
    low_prob = 1;
    for(auto it = word_end_prob.begin(); it != word_end_prob.end(); ) {
        if(it->second == 0) it = word_end_prob.erase(it);
        else {
            it->second = (it->second * 100) / sym_word_count;
            low_prob = std::min(it->second, low_prob);
            it++;
        }
    }
    
    std::vector<std::vector<double>> feature, new_append_feature;
    std::vector<std::string> feature_name, new_append_name;
    
    std::tie(feature, feature_name) = continous_feature(sentences);
    std::tie(new_append_feature, new_append_name) = discrete_feature(sentences);
    append_feature(feature, feature_name, new_append_feature, new_append_name);
    
    return std::make_tuple(feature, res, feature_name);
}

std::pair<std::vector<std::vector<double>>, std::vector<std::string>> FeatureDecoder::continous_feature(std::vector<Sentence>& sentences) {
    std::vector<double> length_of_word;
    std::vector<double> prob_word_end;
    
    for(auto curr : sentences) {
        size_t i = 0;
        for(auto p : curr.words) {
            if(is_symbol(p.first)) {
                length_of_word.push_back(!i ? 0 : curr.words[i - 1].first.size());
                if(!i || !word_end_prob.count(curr.words[i - 1].first)) prob_word_end.push_back(low_prob);
                else prob_word_end.push_back(word_end_prob.at(curr.words[i - 1].first));
            }
            i++;
        }
    }
    
    std::vector<std::string> feature_name = {"length of word"};
    std::vector<std::vector<double>> feature = {length_of_word};
    
    return std::make_pair(feature, feature_name);
}

std::pair<std::vector<std::vector<double>>, std::vector<std::string>> FeatureDecoder::discrete_feature(std::vector<Sentence>& sentences) {
    std::unordered_map<std::string, std::vector<double>> symbol_to_feature;
    std::unordered_map<std::string, std::vector<double>> prev_word_characteristic;
    std::unordered_map<std::string, std::vector<double>> next_word_characteristic;
    std::unordered_map<std::string, std::vector<double>> curr_word_characteristic;
    
    size_t poss_count = 0;
    bool in_quotes = false;
    for(size_t sen_idx = 0; sen_idx < sentences.size(); sen_idx++) {
        Sentence & curr = sentences[sen_idx];
        Sentence * prev = sen_idx ? &sentences[sen_idx] : nullptr;
        size_t i = 0;
        for(auto p : curr.words) {
            if(is_symbol(p.first)) {
                if(symbol_to_feature[p.first].size() != poss_count)
                    symbol_to_feature[p.first].resize(poss_count, false);
                symbol_to_feature[p.first].push_back(true);
                symbol.insert(p.first);
                
                if(is_open_quote(p.first)) {
                    in_quotes = false;
                    curr_word_characteristic["in quotes"].push_back(in_quotes);
                    curr_word_characteristic["open quote"].push_back(true);
                    curr_word_characteristic["close quote"].push_back(false);
                    in_quotes = true;
                } else if(is_close_quote(p.first)) {
                    in_quotes = true;
                    curr_word_characteristic["in quotes"].push_back(in_quotes);
                    curr_word_characteristic["open quote"].push_back(false);
                    curr_word_characteristic["close quote"].push_back(true);
                    in_quotes = false;
                } else {
                    curr_word_characteristic["in quotes"].push_back(in_quotes);
                    curr_word_characteristic["open quote"].push_back(false);
                    curr_word_characteristic["close quote"].push_back(false);
                }
                
                
                if(i || prev) {
                    std::string const & word = i ? curr.words[i - 1].first : prev->words.back().first;
                    prev_word_characteristic["capitalized"].push_back(is_cap(word));
                    prev_word_characteristic["contain symbol"].push_back(contain_symbol(word));
                    prev_word_characteristic["contain number"].push_back(contain_number(word));
                    prev_word_characteristic["is symbol"].push_back(is_symbol(word));
                    prev_word_characteristic["is upper"].push_back(is_upper(word));
                    prev_word_characteristic["no prev"].push_back(false);
                } else {
                    prev_word_characteristic["capitalized"].push_back(false);
                    prev_word_characteristic["contain symbol"].push_back(false);
                    prev_word_characteristic["contain number"].push_back(false);
                    prev_word_characteristic["is symbol"].push_back(false);
                    prev_word_characteristic["is upper"].push_back(false);
                    prev_word_characteristic["no prev"].push_back(true);
                }
                
                if(i != curr.words.size() - 1 || sen_idx != sentences.size() - 1) {
                    std::string const & word = i != curr.words.size() - 1 ? curr.words[i + 1].first : sentences[sen_idx + 1].words[0].first;
                    next_word_characteristic["next capitalized"].push_back(is_cap(word));
                    next_word_characteristic["next contain symbol"].push_back(contain_symbol(word));
                    next_word_characteristic["next contain number"].push_back(contain_number(word));
                    next_word_characteristic["next is open quote"].push_back(is_open_quote(word));
                    next_word_characteristic["next is close quote"].push_back(is_close_quote(word));
                    next_word_characteristic["next is symbol"].push_back(is_symbol(word));
                    next_word_characteristic["next is upper"].push_back(is_upper(word));
                    next_word_characteristic["no next"].push_back(false);
                } else {
                    next_word_characteristic["next capitalized"].push_back(false);
                    next_word_characteristic["next contain symbol"].push_back(false);
                    next_word_characteristic["next contain number"].push_back(false);
                    next_word_characteristic["next is open quote"].push_back(false);
                    next_word_characteristic["next is close quote"].push_back(false);
                    next_word_characteristic["next is symbol"].push_back(false);
                    next_word_characteristic["next is upper"].push_back(false);
                    next_word_characteristic["no next"].push_back(true);
                }
                i++;
                poss_count++;
            }
        }
        
    }
    
    for(auto it = symbol_to_feature.begin(); it != symbol_to_feature.end(); it++)
        it->second.resize(poss_count, false);
    
    
    std::vector<std::vector<double>> feature;
    std::vector<std::string> feature_name;
    
    map_expand(symbol_to_feature, feature, feature_name);
    map_expand(prev_word_characteristic, feature, feature_name);
    map_expand(next_word_characteristic, feature, feature_name);
    map_expand(curr_word_characteristic, feature, feature_name);
    
    return std::make_pair(std::move(feature), std::move(feature_name));
    
}

void FeatureDecoder::map_expand(std::unordered_map<std::string, std::vector<double>> & feature_map, std::vector<std::vector<double>> & feature, std::vector<std::string> & feature_name) {
    for(auto it = feature_map.begin(); it != feature_map.end(); it++) {
        feature.push_back(std::move(it->second));
        feature_name.push_back(std::move(it->first));
    }
}

void FeatureDecoder::append_feature(std::vector<std::vector<double>> & feature, std::vector<std::string> & feature_name, std::vector<std::vector<double>> & new_feature, std::vector<std::string> & new_feature_name) {
    for(size_t i = 0; i < new_feature.size(); i++) {
        feature.push_back(std::move(new_feature[i]));
        feature_name.push_back(std::move(new_feature_name[i]));
    }
}


std::pair<std::vector<std::vector<double>>, std::vector<std::string>> FeatureDecoder::continous_feature(std::vector<std::string>& sentences) {
    std::vector<double> length_of_word;
    std::vector<double> prob_word_end;
    
    size_t i = 0;
    for(auto & p : sentences) {
        if(is_symbol(p)) {
            length_of_word.push_back(!i ? 0 : sentences[i - 1].size());
            if(!i || !word_end_prob.count(sentences[i - 1])) prob_word_end.push_back(low_prob);
            else prob_word_end.push_back(word_end_prob.at(sentences[i - 1]));
        }
        i++;
    }
    
    std::vector<std::string> feature_name = {"length of word"};
    std::vector<std::vector<double>> feature = {length_of_word};
    
    return std::make_pair(feature, feature_name);
}

std::pair<std::vector<std::vector<double>>, std::vector<std::string>> FeatureDecoder::discrete_feature(std::vector<std::string>& sentences) {
    std::unordered_map<std::string, std::vector<double>> symbol_to_feature;
    for(auto it = symbol.begin(); it != symbol.end(); it++)
        symbol_to_feature[*it] = {};
    
    std::unordered_map<std::string, std::vector<double>> prev_word_characteristic;
    std::unordered_map<std::string, std::vector<double>> next_word_characteristic;
    std::unordered_map<std::string, std::vector<double>> curr_word_characteristic;
    
    size_t poss_count = 0;
    size_t i = 0;
    bool in_quotes = false;
    for(auto p : sentences) {
        if(is_symbol(p)) {
            if(symbol_to_feature[p].size() != poss_count)
                symbol_to_feature[p].resize(poss_count, false);
            symbol_to_feature[p].push_back(true);
            
            
            
            if(is_open_quote(p)) {
                in_quotes = false;
                curr_word_characteristic["in quotes"].push_back(in_quotes);
                curr_word_characteristic["open quote"].push_back(true);
                curr_word_characteristic["close quote"].push_back(false);
                in_quotes = true;
            } else if(is_close_quote(p)) {
                in_quotes = true;
                curr_word_characteristic["in quotes"].push_back(in_quotes);
                curr_word_characteristic["open quote"].push_back(false);
                curr_word_characteristic["close quote"].push_back(true);
                in_quotes = false;
            } else {
                curr_word_characteristic["in quotes"].push_back(in_quotes);
                curr_word_characteristic["open quote"].push_back(false);
                curr_word_characteristic["close quote"].push_back(false);
            }
            
            if(i) {
                std::string const & word = sentences[i - 1];
                prev_word_characteristic["capitalized"].push_back(is_cap(word));
                prev_word_characteristic["contain symbol"].push_back(contain_symbol(word));
                prev_word_characteristic["contain number"].push_back(contain_number(word));
                prev_word_characteristic["is symbol"].push_back(is_symbol(word));
                prev_word_characteristic["is upper"].push_back(is_upper(word));
                prev_word_characteristic["no prev"].push_back(false);
            } else {
                prev_word_characteristic["capitalized"].push_back(false);
                prev_word_characteristic["contain symbol"].push_back(false);
                prev_word_characteristic["contain number"].push_back(false);
                prev_word_characteristic["is symbol"].push_back(false);
                prev_word_characteristic["is upper"].push_back(false);
                prev_word_characteristic["no prev"].push_back(true);
            }
            
            if(i != sentences.size() - 1) {
                std::string const & word = sentences[i + 1];
                next_word_characteristic["next capitalized"].push_back(is_cap(word));
                next_word_characteristic["next contain symbol"].push_back(contain_symbol(word));
                next_word_characteristic["next contain number"].push_back(contain_number(word));
                next_word_characteristic["next is open quote"].push_back(is_open_quote(word));
                next_word_characteristic["next is close quote"].push_back(is_close_quote(word));
                next_word_characteristic["next is symbol"].push_back(is_symbol(word));
                next_word_characteristic["next is upper"].push_back(is_upper(word));
                next_word_characteristic["no next"].push_back(false);
            } else {
                next_word_characteristic["next capitalized"].push_back(false);
                next_word_characteristic["next contain symbol"].push_back(false);
                next_word_characteristic["next contain number"].push_back(false);
                next_word_characteristic["next is open quote"].push_back(false);
                next_word_characteristic["next is close quote"].push_back(false);
                next_word_characteristic["next is symbol"].push_back(false);
                next_word_characteristic["next is upper"].push_back(false);
                next_word_characteristic["no next"].push_back(true);
            }
            poss_count++;
        }
        i++;

    }
    
    for(auto it = symbol_to_feature.begin(); it != symbol_to_feature.end(); it++)
        it->second.resize(poss_count, false);
    
    
    std::vector<std::vector<double>> feature;
    std::vector<std::string> feature_name;
    
    map_expand(symbol_to_feature, feature, feature_name);
    map_expand(prev_word_characteristic, feature, feature_name);
    map_expand(next_word_characteristic, feature, feature_name);
    map_expand(curr_word_characteristic, feature, feature_name);
    
    return std::make_pair(std::move(feature), std::move(feature_name));
}

std::vector<std::string> WSJDecoder::read_file_token(std::string file_path) {
    std::fstream file(file_path);
    if(!file.is_open()) throw std::invalid_argument("File: " + file_path + " failed to open");
    
    std::vector<std::string> sentences;
    std::string line;
    while(std::getline(file, line)) {
        if(line.empty()) continue;
        sentences.push_back(line);
    }
    file.close();
    return sentences;
}

std::pair<std::vector<std::vector<double>>, std::unordered_map<std::string, size_t>> WSJDecoder::extract_feature(std::vector<std::string> & sentences) {
    std::vector<std::vector<double>> feature, new_append_feature;
    std::unordered_map<std::string, size_t> feature_name_to_idx;
    std::vector<std::string> feature_name, new_append_name;
    
    std::tie(feature, feature_name) = continous_feature(sentences);
    for(size_t i = 0; i < feature.size(); i++) {
        feature_name_to_idx[feature_name[i]] = i;
    }
    std::tie(new_append_feature, new_append_name) = discrete_feature(sentences);
    for(size_t i = 0; i < new_append_feature.size(); i++) {
        feature.push_back(std::move(new_append_feature[i]));
        feature_name_to_idx[new_append_name[i]] = feature.size() - 1;
    }
    
    return std::make_pair(std::move(feature), std::move(feature_name_to_idx));
}
