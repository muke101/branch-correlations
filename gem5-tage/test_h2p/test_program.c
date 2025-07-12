#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>

bool random_condition(float alpha) {
    return ((float)rand() / RAND_MAX) < alpha;
}

void uncorrelated_function() {
    float alpha = 0.5f;
    for (int i = 0; i < 20; i++) {
        if (random_condition(alpha)) {
        } else {
        }
    }
}

int main() {

    srand(time(NULL));

    int choices[] = {5, 10, 15};
    int num_choices = sizeof(choices) / sizeof(choices[0]);

    for (int z = 0; z < 100; z++) {
      int x = 0;
      int N = choices[z % num_choices];
      for (int i = 0; i < 20; ++i) {
          if (i < N) {
          } else {
              x += 1;
          }
      }

      uncorrelated_function();

      for (int j = 0; j < x; ++j) {

      }
    }

    return 0;
}
