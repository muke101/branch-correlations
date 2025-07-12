CXX = gcc
CXXFLAGS = -std=c++17 -Wall -Wextra -I..
LDFLAGS = -lstdc++

TARGETS = tage_test

all: $(TARGETS)

tage_test: tage_test.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGETS)
