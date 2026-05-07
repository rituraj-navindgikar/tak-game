#ifndef utility_functions_c
#define utility_functions_c
#include <iostream>
#include <string>
#include <vector>
#include <utility>

using namespace std;

int return_digit(char input) {
  if (input >= '0' && input <= '9')
    return (int) (input - '0');
  else
    return -1;
}

void print_vector(vector<pair<int, char>> vec) {
  auto iter = vec.begin();
  cerr << "vector is : ";
  while (iter != vec.end()) {
    cerr << (*iter).first << (*iter).second << " ";
    iter++;
  }
  cerr << endl;
}

#endif
