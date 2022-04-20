#ifndef feature_decoder_hpp
#define feature_decoder_hpp

#include "common.hpp"

struct Sentence {
    std::vector<std::pair<std::string, std::string>> words;
};

class FeatureDecoder {
protected:
    
    std::unordered_map<std::string, double> word_end_prob;
    double low_prob = 1;
    std::unordered_set<std::string> symbol;
    
    virtual std::pair<std::vector<std::vector<double>>, std::vector<std::string>> continous_feature(std::vector<Sentence>& sentences);
    virtual std::pair<std::vector<std::vector<double>>, std::vector<std::string>> continous_feature(std::vector<std::string>& sentences);
    
    virtual std::pair<std::vector<std::vector<double>>, std::vector<std::string>> discrete_feature(std::vector<Sentence>& sentences);
    virtual std::pair<std::vector<std::vector<double>>, std::vector<std::string>> discrete_feature(std::vector<std::string>& sentences);
    
    void map_expand(std::unordered_map<std::string, std::vector<double>> & feature_map, std::vector<std::vector<double>> & feature, std::vector<std::string> & feature_name);
    
    void append_feature(std::vector<std::vector<double>> & feature, std::vector<std::string> & feature_name, std::vector<std::vector<double>> & new_feature, std::vector<std::string> & new_feature_name);
public:
    std::string raw_read(std::string file_path);
    virtual std::vector<std::string> read_file_token(std::string file_path)=0;
    virtual std::vector<Sentence> read_file(std::string file_path)=0;
    virtual std::tuple<std::vector<std::vector<double>>, std::vector<bool>, std::vector<std::string>> feature_decode(std::vector<Sentence>& sentences)=0;
    virtual std::pair<std::vector<std::vector<double>>, std::unordered_map<std::string, size_t>> extract_feature(std::vector<std::string> & sentences)=0;
};

class WSJDecoder : public FeatureDecoder {

public:
    std::vector<Sentence> read_file(std::string file_path) override;
    std::vector<std::string> read_file_token(std::string file_path) override;
    std::tuple<std::vector<std::vector<double>>, std::vector<bool>, std::vector<std::string>> feature_decode(std::vector<Sentence>& head) override;
    std::pair<std::vector<std::vector<double>>, std::unordered_map<std::string, size_t>> extract_feature(std::vector<std::string> & sentences) override;
};

#endif /* feature_decoder_hpp */
