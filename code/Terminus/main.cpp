#include "common.hpp"
#include "util.hpp"
#include "adaboosting.hpp"
#include "feature_decoder.hpp"
#include "json.hpp"

using namespace std;

void print_res(vector<bool> & prediction, vector<bool> & correct_target, std::vector<std::string> & sentences) {
    size_t tn = 0, tp = 0, fn = 0, fp = 0;
    for(size_t i = 0; i < prediction.size(); i++) {
        if(prediction[i] && correct_target[i]) tp++;
        else if(prediction[i] && !correct_target[i]) fp++;
        else if(!prediction[i] && correct_target[i]) fn++;
        else tn++;
    }
    
    vector<size_t> feature_loc;
    for(size_t i = 0; i < sentences.size(); i++) {
        if(is_symbol(sentences[i])) {
            feature_loc.push_back(i);
        }
    }
    
    for(size_t i = 0; i < prediction.size(); i++) {
        if(prediction[i] && !correct_target[i]) {
            cout << "false positive: " << sentences[feature_loc[i]] << " " << i << endl;
            size_t start = feature_loc[i] >= 5 ? feature_loc[i] - 5 : 0;
            size_t end = feature_loc[i] + 5 >= sentences.size() ? sentences.size() : feature_loc[i] + 5;
            for(; start < end; start++) cout << sentences[start] << " ";
            
            cout << endl << endl;
        }
        else if(!prediction[i] && correct_target[i]) {
            cout << "false negative: " << sentences[feature_loc[i]] << " " << i  << endl;
            size_t start = feature_loc[i] >= 5 ? feature_loc[i] - 5 : 0;
            size_t end = feature_loc[i] + 5 >= sentences.size() ? sentences.size() : feature_loc[i] + 5;
            for(; start < end; start++) cout << sentences[start] << " ";
            
            cout << endl << endl;
        }
    }
    
    cout << "true negative = " << tn << endl;
    cout << "true positive = " << tp << endl;
    cout << "false negative = " << fn << endl;
    cout << "false postive = " << fp << endl;
    
    double recall = (tp * 1.0) / (tp + fp * 1.0), precision = (tp * 1.0) / (tp + fn * 1.0);
    
    cout << "recall = " << recall << endl;
    cout << "precision = " << precision << endl;
    cout << "f-score = " << 2 * (precision * recall) / (precision + recall) << endl;
    cout << "Accuracy = " << (tp + tn * 1.0) / (tp + tn + fp + fn * 1.0) << endl;
}

int main(int argc, const char * argv[]) {
    WSJDecoder decoder;
    
    vector<Sentence> head = decoder.read_file("/Users/morgan/Desktop/Terminus/datasets/WSJ_POS_CORPUS_FOR_STUDENTS/WSJ_02-21.pos");
    std::vector<std::vector<double>> feature;
    std::vector<bool> target;
    std::vector<std::string> feature_name;
    std::tie(feature, target, feature_name) = decoder.feature_decode(head, "/Users/morgan/Desktop/Terminus/code/Terminus/terminus_unigram/uni.json");
    
    std::vector<std::vector<double>> predict_feature;
    phmap::flat_hash_map<std::string, size_t> feature_name_to_idx;
    std::vector<std::string> sentences = decoder.read_file_token("/Users/morgan/Desktop/Terminus/datasets/WSJ_POS_CORPUS_FOR_STUDENTS/WSJ_24.words");
    tie(predict_feature, feature_name_to_idx) = decoder.extract_feature(sentences, "/Users/morgan/Desktop/Terminus/code/Terminus/terminus_unigram/uni.json");
    
    BinaryAdaBoosting booster;
    booster.fit(feature, target, 100, feature_name, 1);

    vector<bool> prediction = booster.predict(predict_feature, feature_name_to_idx);
    
    std::vector<std::vector<double>> correct_feature;
    std::vector<bool> correct_target;
    head = decoder.read_file("/Users/morgan/Desktop/Terminus/datasets/WSJ_POS_CORPUS_FOR_STUDENTS/WSJ_24.pos");
    std::tie(correct_feature, correct_target, feature_name) = decoder.feature_decode(head, "/Users/morgan/Desktop/Terminus/code/Terminus/terminus_unigram/uni.json");
    
    print_res(prediction, correct_target, sentences);

    return 0;
}
