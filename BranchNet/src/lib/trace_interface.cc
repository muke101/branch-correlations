#include "trace_interface.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>

std::vector<HistElt> read_trace(char* input_trace) {
  std::vector<HistElt> history;

  const int BUFFER_SIZE = 4096;
  char      cmd[BUFFER_SIZE];

  auto out = snprintf(cmd, BUFFER_SIZE, "bzip2 -dc %s", input_trace);
  assert(out < BUFFER_SIZE);

  FILE*   fptr = popen(cmd, "r");
  HistElt history_elt_buffer;
  while (fread(&history_elt_buffer, sizeof(history_elt_buffer), 1, fptr) == 1) {
    history.push_back(history_elt_buffer);
  }

  pclose(fptr);

  return history;
}
