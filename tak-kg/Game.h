#ifndef game_h
#define game_h

#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>
#include <algorithm>
#include <utility>
#include <ctype.h>

#include "utility_functions.cpp"

using namespace std;

struct Player{
  int flats;
  int capstones;
};

class Game{
public:
  int n;
  int total_squares;
  int max_movable;
  int max_up;
  int max_down;
  char max_left;
  char max_right;
  vector<string> all_squares;

  Game() {}

  void initialize_game(int n);

  int square_to_num(string square_string);

  string square_to_string(int square);

  vector<std::vector<int>> partition(int n);

  bool check_valid(int, char, vector<int>, vector<vector<pair<int, char>>>);

  vector<string> generate_stack_moves(int square, vector<vector<pair<int, char>>>);

  vector<string> generate_all_moves(int, bool, vector<vector<pair<int, char>>>, vector<Player>);

  vector<vector<pair<int, char>>> execute_move(int, string, vector<vector<pair<int, char>>>, vector<Player> *);

};
#endif
