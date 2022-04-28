#include "feature_decoder.hpp"
#include "util.hpp"

nlohmann::json FeatureDecoder::read_dictionary(std::string const & filepath) {
    return nlohmann::json::parse(raw_read(filepath));
}

std::string FeatureDecoder::raw_read(std::string const & filepath) {
    std::fstream file(filepath);
    if(!file.is_open()) throw std::invalid_argument("File: " + filepath + " failed to open");
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

std::tuple<std::vector<std::vector<double>>, std::vector<bool>, std::vector<std::string>> WSJDecoder::feature_decode(std::vector<Sentence>& sentences, std::string const & dpath) {
    std::vector<bool> res;
    
    
    for(auto curr : sentences) {
        size_t i = 0;
        for(auto p : curr.words) {
            if(is_symbol(p.first))
                res.push_back(i == curr.words.size() - 1);
            
            i++;
        }
    }
    
    
    std::vector<std::vector<double>> feature, new_append_feature;
    std::vector<std::string> feature_name, new_append_name;
    
    std::tie(feature, feature_name) = continous_feature(sentences, dpath);
    std::tie(new_append_feature, new_append_name) = discrete_feature(sentences, dpath);
    append_feature(feature, feature_name, new_append_feature, new_append_name);
    
    return std::make_tuple(feature, res, feature_name);
}

std::pair<std::vector<std::vector<double>>, std::vector<std::string>> FeatureDecoder::continous_feature(std::vector<Sentence>& sentences, std::string const & dpath) {
    std::vector<double> length_of_word;
    
    for(auto curr : sentences) {
        size_t i = 0;
        for(auto p : curr.words) {
            if(is_symbol(p.first)) {
                if(i) {
                    length_of_word.push_back(curr.words[i - 1].first.size());
                } else {
                    length_of_word.push_back(0);
                    
                }
                
            }
            i++;
        }
    }
    
    std::vector<std::string> feature_name = {"length of word"};
    std::vector<std::vector<double>> feature = {length_of_word};
    
    
    return std::make_pair(feature, feature_name);
}

std::pair<std::vector<std::vector<double>>, std::vector<std::string>> FeatureDecoder::discrete_feature(std::vector<Sentence>& sentences, std::string const & dpath) {
    phmap::flat_hash_map<std::string, std::vector<double>> symbol_to_feature;
    phmap::flat_hash_map<std::string, std::vector<double>> prev_word_characteristic;
    phmap::flat_hash_map<std::string, std::vector<double>> next_word_characteristic;
    phmap::flat_hash_map<std::string, std::vector<double>> curr_word_characteristic;
    phmap::flat_hash_map<std::string, std::vector<double>> word_pos_prob;
    phmap::flat_hash_map<std::string, std::vector<double>> next_word_pos_prob;
    
    nlohmann::json dic = read_dictionary(dpath);
    word_to_prob.clear();
    
    for(auto it = dic.begin(); it != dic.end(); it++) {
        nlohmann::json v = it.value();
        std::string key = it.key();
        
        std::vector<std::string> pos;
        
        if(is_symbol(key) || key.compare("SYM") == 0) {
            pos.push_back(key);
        }
        if(key.compare("NPS") == 0 || key.compare("NNP") == 0) {
            pos.push_back("proper noun");
        }
        if(key.compare("PRP") == 0 || key.compare("PRP$") == 0 || key.compare("WP") == 0) {
            pos.push_back("pronoun");
        }
        if(key.compare("JJR") == 0 || key.compare("JJS") || key.compare("JJ") == 0 || key.compare("JJ|RB") == 0 || key.compare("RBR") == 0 || key.compare("RBS") == 0 || key.compare("WRB") == 0) {
            pos.push_back("modifier");
        }
        if(key.compare("VBP|VBD") == 0 || key.compare("VBG") == 0 || key.compare("VBN") == 0 || key.compare("VBZ") == 0 || key.compare("VBD") == 0 || key.compare("VBG|NN") == 0) {
            pos.push_back("verb");
        }
        if(key.compare("PDT") == 0 || key.compare("DT") == 0) {
            pos.push_back("determiner");
        }
        if(key.compare("CD|NN") == 0 || key.compare("NN") == 0 || key.compare("CD|NNS") == 0 || key.compare("CD|NN||NP") == 0 || key.compare("VBG|NN") == 0) {
            pos.push_back("noun");
        }
        if(pos.empty()) pos.push_back("other");
        
        for(std::string const & p : pos) {
            for(auto it2 = v.begin(); it2 != v.end(); it2++) {
                word_to_prob[it2.key()][p] = 1;
            }
        }
    }
    
    size_t poss_count = 0;
    bool in_quotes = false;
    for(size_t sen_idx = 0; sen_idx < sentences.size(); sen_idx++) {
        Sentence & curr = sentences[sen_idx];
        Sentence * prev = sen_idx ? &sentences[sen_idx] : nullptr;
        size_t i = 0;
        for(auto p : curr.words) {
            if(is_symbol(p.first)) {
                if(!is_quote(p.first)) {
                    if(symbol_to_feature[p.first].size() != poss_count)
                        symbol_to_feature[p.first].resize(poss_count, false);
                    symbol_to_feature[p.first].push_back(true);
                    symbol.insert(p.first);
                }
                
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
                    
                    if(word_to_prob.count(word)) {
                        for(auto const & it : word_to_prob[word]) {
                            word_pos_prob["prev " + it.first].resize(poss_count, 0);
                            word_pos_prob["prev " + it.first].push_back(it.second);
                        }
                    } else {
                        word_pos_prob["prev OOV"].resize(poss_count, 0);
                        word_pos_prob["prev OOV"].push_back(1);
                    }
                    
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
                    
                    if(word_to_prob.count(word)) {
                        for(auto const & it : word_to_prob[word]) {
                            next_word_pos_prob["next " + it.first].resize(poss_count, 0);
                            next_word_pos_prob["next " + it.first].push_back(it.second);
                        }
                    } else {
                        next_word_pos_prob["next OOV"].resize(poss_count, 0);
                        next_word_pos_prob["next OOV"].push_back(1);
                    }
                    
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
    
    for(auto const & it : word_pos_prob)
        word_pos_prob[it.first].resize(poss_count, 0);
    word_pos_prob["prev OOV"].resize(poss_count, 0);
    
    for(auto const & it : next_word_pos_prob)
        next_word_pos_prob[it.first].resize(poss_count, 0);
    next_word_pos_prob["next OOV"].resize(poss_count, 0);
    
    
    for(auto it = symbol_to_feature.begin(); it != symbol_to_feature.end(); it++)
        it->second.resize(poss_count, false);
    
    
    std::vector<std::vector<double>> feature;
    std::vector<std::string> feature_name;
    
    map_expand(symbol_to_feature, feature, feature_name);
    map_expand(prev_word_characteristic, feature, feature_name);
    map_expand(next_word_characteristic, feature, feature_name);
    map_expand(curr_word_characteristic, feature, feature_name);
    map_expand(word_pos_prob, feature, feature_name);
    map_expand(next_word_pos_prob, feature, feature_name);
    
    return std::make_pair(std::move(feature), std::move(feature_name));
    
}

void FeatureDecoder::map_expand(phmap::flat_hash_map<std::string, std::vector<double>> & feature_map, std::vector<std::vector<double>> & feature, std::vector<std::string> & feature_name) {
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


std::pair<std::vector<std::vector<double>>, std::vector<std::string>> FeatureDecoder::continous_feature(std::vector<std::string>& sentences, std::string const & dpath) {
    std::vector<double> length_of_word;
    
    size_t i = 0;
    for(auto & p : sentences) {
        if(is_symbol(p)) {
            if(i) {
                length_of_word.push_back(sentences[i - 1].size());
            } else {
                length_of_word.push_back(0);
            }
            
        }
        i++;
    }
    
    std::vector<std::string> feature_name = {"length of word"};
    std::vector<std::vector<double>> feature = {length_of_word};
    
    return std::make_pair(feature, feature_name);
}

std::pair<std::vector<std::vector<double>>, std::vector<std::string>> FeatureDecoder::discrete_feature(std::vector<std::string>& sentences) {
    phmap::flat_hash_map<std::string, std::vector<double>> symbol_to_feature;
    for(auto it = symbol.begin(); it != symbol.end(); it++)
        symbol_to_feature[*it] = {};
    
    phmap::flat_hash_map<std::string, std::vector<double>> prev_word_characteristic;
    phmap::flat_hash_map<std::string, std::vector<double>> next_word_characteristic;
    phmap::flat_hash_map<std::string, std::vector<double>> curr_word_characteristic;
    phmap::flat_hash_map<std::string, std::vector<double>> word_pos_prob;
    phmap::flat_hash_map<std::string, std::vector<double>> next_word_pos_prob;
    
    size_t poss_count = 0;
    size_t i = 0;
    bool in_quotes = false;
    for(auto p : sentences) {
        if(is_symbol(p)) {
            if(!is_quote(p)) {
                if(symbol_to_feature[p].size() != poss_count)
                    symbol_to_feature[p].resize(poss_count, false);
                symbol_to_feature[p].push_back(true);
            }
            
            
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
                
                if(word_to_prob.count(word)) {
                    for(auto const & it : word_to_prob[word]) {
                        word_pos_prob["prev " + it.first].resize(poss_count, 0);
                        word_pos_prob["prev " + it.first].push_back(it.second);
                    }
                } else {
                    word_pos_prob["prev OOV"].resize(poss_count, 0);
                    word_pos_prob["prev OOV"].push_back(1);
                }
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
                
                if(word_to_prob.count(word)) {
                    for(auto const & it : word_to_prob[word]) {
                        next_word_pos_prob["next " + it.first].resize(poss_count, 0);
                        next_word_pos_prob["next " + it.first].push_back(it.second);
                    }
                } else {
                    next_word_pos_prob["next OOV"].resize(poss_count, 0);
                    next_word_pos_prob["next OOV"].push_back(1);
                }
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
    
    for(auto const & it : word_pos_prob)
        word_pos_prob[it.first].resize(poss_count, 0);
    word_pos_prob["prev OOV"].resize(poss_count, 0);
    
    for(auto const & it : next_word_pos_prob)
        next_word_pos_prob[it.first].resize(poss_count, 0);
    next_word_pos_prob["next OOV"].resize(poss_count, 0);
    
    for(auto it = symbol_to_feature.begin(); it != symbol_to_feature.end(); it++)
        it->second.resize(poss_count, false);
    
    
    std::vector<std::vector<double>> feature;
    std::vector<std::string> feature_name;
    
    map_expand(symbol_to_feature, feature, feature_name);
    map_expand(prev_word_characteristic, feature, feature_name);
    map_expand(next_word_characteristic, feature, feature_name);
    map_expand(curr_word_characteristic, feature, feature_name);
    map_expand(word_pos_prob, feature, feature_name);
    map_expand(next_word_pos_prob, feature, feature_name);
    
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

std::pair<std::vector<std::vector<double>>, phmap::flat_hash_map<std::string, size_t>> WSJDecoder::extract_feature(std::vector<std::string> & sentences, std::string const & dpath) {
    std::vector<std::vector<double>> feature, new_append_feature;
    phmap::flat_hash_map<std::string, size_t> feature_name_to_idx;
    std::vector<std::string> feature_name, new_append_name;
    
    std::tie(feature, feature_name) = continous_feature(sentences, dpath);
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
