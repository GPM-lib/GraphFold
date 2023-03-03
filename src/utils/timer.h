#ifndef UTILS_TIMER_H
#define UTILS_TIMER_H

#include <sys/time.h>

#include <random>

namespace project_GraphFold {

inline double wtime() {
  struct timeval time1;
  gettimeofday(&time1, NULL);
  return time1.tv_sec + time1.tv_usec * 1e-6;
}

}  // namespace project_GraphFold

#endif  // UTILS_TIMER_H
