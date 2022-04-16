#include "common.hpp"
#include "util.hpp"
#include "adaboosting.hpp"

using namespace std;

// TODO: Test all the methods

int main(int argc, const char * argv[]) {
    vector<vector<double>> data{
        {0,1,1,1,0,1,0,0,1,1,0},
        {35,5,20,30,6,15,40,70,45,15,80},
    };
    vector<bool> y{0,0,1,1,0,0,0,0,1,0,0};
    
    BinaryAdaBoosting b;
    b.fit(data, y, 3);
    vector<bool> t = b.predict({{0,1,1}, {75, 5, 40}});
    return 0;
}
