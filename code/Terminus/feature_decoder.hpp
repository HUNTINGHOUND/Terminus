#ifndef feature_decoder_hpp
#define feature_decoder_hpp

#include "common.hpp"
#include "json.hpp"

struct Sentence {
    std::vector<std::pair<std::string, std::string>> words;
};

class FeatureDecoder {
protected:
    
    phmap::flat_hash_map<std::string, phmap::flat_hash_map<std::string, double>> word_to_prob;
    
    double low_prob = 1;
    phmap::flat_hash_set<std::string> symbol;
    
    virtual std::pair<std::vector<std::vector<double>>, std::vector<std::string>> continous_feature(std::vector<Sentence>& sentences, std::string const & dpath);
    virtual std::pair<std::vector<std::vector<double>>, std::vector<std::string>> continous_feature(std::vector<std::string>& sentences, std::string const & dpath);
    
    virtual std::pair<std::vector<std::vector<double>>, std::vector<std::string>> discrete_feature(std::vector<Sentence>& sentences, std::string const & dpath);
    virtual std::pair<std::vector<std::vector<double>>, std::vector<std::string>> discrete_feature(std::vector<std::string>& sentences);
    
    void map_expand(phmap::flat_hash_map<std::string, std::vector<double>> & feature_map, std::vector<std::vector<double>> & feature, std::vector<std::string> & feature_name);
    
    void append_feature(std::vector<std::vector<double>> & feature, std::vector<std::string> & feature_name, std::vector<std::vector<double>> & new_feature, std::vector<std::string> & new_feature_name);
    
    nlohmann::json read_dictionary(std::string const & filepath);
public:
    std::string raw_read(std::string const & filepath);
    virtual std::vector<std::string> read_file_token(std::string file_path)=0;
    virtual std::vector<Sentence> read_file(std::string file_path)=0;
    virtual std::tuple<std::vector<std::vector<double>>, std::vector<bool>, std::vector<std::string>> feature_decode(std::vector<Sentence>& sentences, std::string const & dpath)=0;
    virtual std::pair<std::vector<std::vector<double>>, phmap::flat_hash_map<std::string, size_t>> extract_feature(std::vector<std::string> & sentences, std::string const & dpath)=0;
};

class WSJDecoder : public FeatureDecoder {

public:
    std::vector<Sentence> read_file(std::string file_path) override;
    std::vector<std::string> read_file_token(std::string file_path) override;
    std::tuple<std::vector<std::vector<double>>, std::vector<bool>, std::vector<std::string>> feature_decode(std::vector<Sentence>& head, std::string const & dpath) override;
    std::pair<std::vector<std::vector<double>>, phmap::flat_hash_map<std::string, size_t>> extract_feature(std::vector<std::string> & sentences, std::string const & dpath) override;
};

#endif /* feature_decoder_hpp */
