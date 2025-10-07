#include "trace_interface.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>

FILE* open_trace(char* input_trace) {
  const int BUFFER_SIZE = 4096;
  char      cmd[BUFFER_SIZE];

  auto out = snprintf(cmd, BUFFER_SIZE, "bzip2 -dc %s", input_trace);
  assert(out < BUFFER_SIZE);

  FILE* fptr = popen(cmd, "r");

  return fptr;
}

std::vector<HistElt> read_trace(FILE* input_trace, size_t chunk_size) {
  std::vector<HistElt> history;

  HistElt history_elt_buffer;
  for (int i = 0; i < chunk_size && fread(&history_elt_buffer, sizeof(history_elt_buffer), 1, input_trace) == 1; i++) {
    history.push_back(history_elt_buffer);
  }

  return history;
}
