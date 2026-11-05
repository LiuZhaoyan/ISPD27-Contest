// Small C ABI around the unmodified external FLUTE implementation.
// FLUTE: Dr. Chris C. N. Chu, Iowa State University,
// http://class.ee.iastate.edu/cnchu/ (see the external source LICENSE).
#include "flute.hpp"
#include <cstdlib>
extern "C" {
void quad_flute_init(const char* powv, const char* post) {
  flute::readLUT(powv, post);
}
int quad_flute_tree(int degree, int* x, int* y, int* output) {
  if (degree < 2 || degree >= MAXD) return -1;
  auto tree = flute::flute(degree, x, y, 3);
  int n = 2 * degree - 2;
  for (int i = 0; i < n; ++i) {
    output[3*i] = tree.branch[i].x;
    output[3*i+1] = tree.branch[i].y;
    output[3*i+2] = tree.branch[i].n;
  }
  std::free(tree.branch);
  return n;
}
}
