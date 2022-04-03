#ifndef util_hpp
#define util_hpp

#include "common.hpp"

// Template programming utils

template <typename T>
struct is_built_in : std::bool_constant<std::is_integral<T>::value || std::is_floating_point<T>::value>{};

template <typename X>
inline constexpr auto is_built_in_v = is_built_in<X>::value;

// Vector utils

template <typename T, typename S>
std::vector<bool> not_equal(std::vector<T> const & a, std::vector<S> const & b) {
    static_assert(is_built_in_v<T> && is_built_in_v<S>, "Vector type must be built in");
    if(a.size() != b.size()) throw std::invalid_argument("Vector size are not equal");
    
    std::vector<bool> res(a.size());
    for(size_t i = 0; i < std::min(a.size(), b.size()); i++) res[i] = a[i] != b[i];
    
    return res;
}

#endif /* util_hpp */
