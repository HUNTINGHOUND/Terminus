#include "util.hpp"

bool is_symbol(std::string const & s) {
    if(s.size() != 1 && !is_quote(s)) return false;
    return !isalnum(s[0]);
}

bool is_cap(std::string const & s) {
    return s.size() != 0 && isupper(s[0]);
}

bool is_upper(std::string const & s) {
    for(size_t i = 0; i < s.size(); i++)
        if(!isupper(s[i])) return false;
    return true;
}

bool contain_number(std::string const & s) {
    for(size_t i = 0; i < s.size(); i++)
        if(isnumber(s[i])) return true;
    return false;
}

bool contain_symbol(std::string const & s) {
    for(size_t i = 0; i < s.size(); i++)
        if(!isalnum(s[i])) return true;
    return false;
}

bool is_quote(std::string const & s) {
    return s.compare("''") == 0 || s.compare("``") == 0 || s.compare("\"") == 0;
}

bool is_open_quote(std::string const & s) {
    return s.compare("``") == 0;
}

bool is_close_quote(std::string const & s) {
    return s.compare("''") == 0;
}
