#ifndef UTILS_MACROS_H
#define UTILS_MACROS_H

#include <cstdlib>
#include <iostream>

namespace project_GraphFold {

struct Empty {
  char placeholder;
};

std::istream& operator>>(std::istream& in, Empty& e) {
  in >> e.placeholder;
  return in;
}

std::ostream& operator<<(std::ostream& out, Empty& e) {
  out << e.placeholder;
  return out;
}

enum algorithms { TC, SM, CF, MC, FSM };
enum modes { v_centric, e_centric };
typedef unsigned long long AccType;

#ifndef CEIL_DIV
#define CEIL_DIV(a, b) (((b) != 0) ? ((a) + (b) - 1) / (b) : 0)
#endif

}  // namespace project_GraphFold

#endif  // UTILS_MACROS_H
