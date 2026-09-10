#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <vector>
using Tuple = std::array<size_t, 2>;
int main() {
  std::cout << "size,method,sample,seconds,comparisons\n";
  for (size_t size : {1000, 4000, 8000}) {
    std::vector<Tuple> a, b;
    for (size_t i = 0; i < size; ++i) a.push_back({i, i+1});
    b = a;
    std::ranges::reverse(b);
    for (bool sorted : {false, true}) {
      for (int sample = 0; sample < 7; ++sample) {
        size_t comparisons = 0;
        auto start = std::chrono::steady_clock::now();
        bool same;
        if (sorted) {
          std::vector<const Tuple*> left, right;
          left.reserve(size); right.reserve(size);
          for (auto& tuple : a) left.push_back(&tuple);
          for (auto& tuple : b) right.push_back(&tuple);
          auto less = [&](auto x, auto y) { ++comparisons; return *x < *y; };
          std::ranges::sort(left, less); std::ranges::sort(right, less);
          same = std::ranges::equal(left,right,[&](auto x,auto y){++comparisons;return *x==*y;});
        } else {
          same = std::ranges::is_permutation(a,b,[&](auto& x,auto& y){++comparisons;return x==y;});
        }
        auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        assert(same);
        std::cout << size << ',' << (sorted ? "sorted_views" : "permutation") << ',' << sample << ',' << elapsed << ',' << comparisons << '\n';
      }
    }
  }
}
