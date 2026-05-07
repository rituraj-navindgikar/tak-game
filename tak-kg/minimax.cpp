#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>
#include <limits>
#include <math.h>
#include <sys/time.h>
#include <random>

#include "Game.h"
#include "Game.cpp"

class features{
public:
  static int diff_flats(vector<vector<pair<int, char>>> state, int player) {
    int counter = 0;
    for (auto iter = state.begin(); iter != state.end(); iter++) {
      if ((*iter).size() > 0) {
        if ((*iter).back().second == 'F') {
          if ((*iter).back().first == player) {
            counter++;
          } else {
            counter--;
          }
        }
      }
    }
    return counter;
  }
};

class minimax{
public:
  int player;
  int n;
  struct timeval start;
  struct timeval finish;
  double time_limit;
  double time_remaining;
  int depth_initial;
  std::vector<vector<pair<int, char>>> state;
  std::vector<Player> players;
  Game game;
  vector<int> x;
  vector<int> y;
  vector<bool> has_left;
  vector<bool> has_right;
  vector<bool> has_down;
  vector<bool> has_up;
  int min_x, max_x, min_y, max_y;

  minimax() {
    int player, n, time_input;
    cin >> player >> n >> time_input;
    this->player = player - 1;
    this->n = n;
    this->time_limit = time_input;
    this->time_remaining = time_input;
    game.initialize_game(this->n);
    this->initialize_flood_fill();
    this->play();
  }

  void play() {
    double alpha_initial = -1 * std::numeric_limits<double>::infinity();
    double beta_initial = std::numeric_limits<double>::infinity();
    int moves = 0;

    for (int i = 0; i < this->game.total_squares; i++) {
      vector<pair<int, char>> initial;
      state.push_back(initial);
    }
    players.resize(2);
    players[0].flats = 21;
    players[0].capstones = 1;

    players[1].flats = 21;
    players[1].capstones = 1;
    while (true) {
      if (moves == 0) {
        if (this->player == 0) {
          gettimeofday(&start, NULL);
          depth_initial = depth(moves);
          double value = maxval(state, players, alpha_initial, beta_initial, depth_initial, true, moves);
          moves++;
        } else {
          string oponent_move;
          cin >> oponent_move;
          gettimeofday(&start, NULL);
          //std::cerr << "Oponent Move : " << oponent_move << std::endl;
          //std::cerr << "Time Remaining : " << time_remaining << std::endl;
          //cerr << "Executing Move" << endl;
          state = this->game.execute_move(this->player, oponent_move, state, &players);
          moves++;
          depth_initial = depth(moves);
          double value = maxval(state, players, alpha_initial, beta_initial, depth_initial, true, moves);
          moves++;
        }
      } else if (moves == 1) {
        string oponent_move;
        cin >> oponent_move;
        gettimeofday(&start, NULL);
        //std::cerr << "Oponent Move : " << oponent_move << std::endl;
        //std::cerr << "Time Remaining : " << time_remaining << std::endl;
        //cerr << "Executing Move" << endl;
        state = this->game.execute_move(this->player, oponent_move, state, &players);
        moves++;
        depth_initial = depth(moves);
        double value = maxval(state, players, alpha_initial, beta_initial, depth_initial, false, moves);
        moves++;
      } else {
        string oponent_move;
        cin >> oponent_move;
        gettimeofday(&start, NULL);
        //std::cerr << "Oponent Move : " << oponent_move << std::endl;
        //std::cerr << "Time Remaining : " << time_remaining << std::endl;
        //cerr << "Executing Move" << endl;
        state = this->game.execute_move(1 - this->player, oponent_move, state, &players);
        moves++;
        //cerr << "Evaluation Function on state" << endl;
        //pair<bool, double> result = evaluation_function1(state, players, moves);
        depth_initial = depth(moves);
        double value = maxval(state, players, alpha_initial, beta_initial, depth_initial, false, moves);
        moves++;
      }
    }
  }
  
  int depth(int moves) {
    int output;
    if (moves < 6 || (this->time_remaining < ((time_limit * 40) / 100) && this->time_remaining > ((time_limit * 5) / 100))) {
      output = 2;
    } else if (this->time_remaining < ((time_limit * 5) / 100)){
      output = 1;
    } else {
		return 3;
	}
    //cerr << "Current Depth : " << output << endl;
    return output;
  }

  void initialize_flood_fill() {
    x.resize(n * n);
    y.resize(n * n);
    has_left.resize(n * n);
    has_right.resize(n * n);
    has_up.resize(n * n);
    has_down.resize(n * n);

    for(int i = 0; i < n; i++){
  		for(int j = 0; j < n; j++){
  			x[i + n*j] = i;
  			y[i + n*j] = j;

  			if (i > 0){
  				has_left[i+n*j] = true;
  			}
  			if( i == 0){
  				has_left[i+n*j] = false;
  			}
  			if (i < (n - 1)){
  				has_right[i+n*j] = true;
  			}
  			if( i == n - 1){
  				has_left[i+n*j] = true;
  			}
        if (j > 0){
  				has_down[i+n*j] = true;
  			}
  			if( j == 0){
  				has_down[i+n*j] = false;
  			}
  			if (j < (n - 1)){
  				has_up[i+n*j] = true;
  			}
  			if( j == (n - 1)){
  				has_up[i+n*j] = false;
  			}
  		}
  	}
  }

  int flood_fill(int j, int player, vector<vector<pair<int, char>>> state, vector<bool>* explored)
  {
  	int sum = 0;
    if ((state[j].size() > 0)) {
    	if (!(*explored)[j] && (state[j].back().first == player) && (state[j].back().second == 'F' || state[j].back().second == 'C'))
    	{
    		(*explored)[j] = true;
    		sum = sum + 1;

    		min_x = std::min(x[j],min_x);
    		max_x = std::max(x[j],max_x);
    		min_y = std::min(y[j],min_y);
    		max_y = std::max(y[j],max_y);

    		if (has_left[j]) {
          if (!(*explored)[j - 1]){
    			  sum += flood_fill(j - 1, player, state, explored);
          }
    		}

    		if (has_right[j]){
          if (!(*explored)[j + 1]) {
    			  sum += flood_fill(j + 1, player, state, explored);
          }
    		}

    		if (has_down[j]){
          if (!(*explored)[j - n]) {
    			  sum += flood_fill(j - n, player, state, explored);
          }
        }

    		if (has_up[j]){
          if (!(*explored)[j + n]) {
            sum += flood_fill(j + n, player, state, explored);
          }
        }
    	}
    }
    return sum;
  }

  bool are_remaining_pieces(vector<Player> players) {
    bool output = false;
    for (int i = 0; i < players.size(); i++) {
      if (players[i].flats > 0 || players[i].capstones > 0) {
        output = true;
      }
    }
    return output;
  }

  double position_strength(int i) {
    double x_strength = abs(((n + 1) / 2) - x[i]);
    double y_strength = abs(((n + 1) / 2) - y[i]);
    return x_strength + y_strength;
  }

  vector<int> surrounding_influencer(int s, int i, int player, vector<vector<pair<int, char>>> state) {
  	vector<int> output(5);
    if (state[i - s].size() > 0) {
      if(state[i-s].back().first == (1- player) && (state[i-s].back().second == 'F' ) ){
    		output[0] = 1;
    	} else if(state[i-s].back().first == (1- player) && (state[i-s].back().second == 'S' || state[i-s].back().second == 'C')) {
    		output[1] = 1;
    	} else if(state[i-s].back().first == (1- player) && state[i-s].back().second == 'C'){
    		output[2] = 1;
    	} else if(state[i-s].back().first == player && state[i-s].back().second == 'F') {
    		output[3] = 1;
    	}else if(state[i-s].back().first == player && (state[i-s].back().second == 'S' || state[i-s].back().second == 'C')) {
    		output[4] = 1;
    	}
    }
  	return output;
  }

  vector<int> rough_influence_measure(int i, int player, vector<vector<pair<int, char>>> state) {
  	int enemy_flats = 0;
  	int enemy_blocks = 0;
  	int enemy_caps = 0;
  	int self_flats = 0;
  	int self_blocks = 0;
  	if (has_left[i]) {
  		vector<int> inf = surrounding_influencer(1, i, player, state);
  		enemy_flats += inf[0];
  		enemy_blocks += inf[1];
  		enemy_caps += inf[2];
  		self_flats += inf[3];
  		self_blocks += inf[4];
  	}
  	if (has_down[i]) {
  		vector<int> inf = surrounding_influencer(n, i, player, state);
  		enemy_flats += inf[0];
  		enemy_blocks += inf[1];
  		enemy_caps += inf[2];
  		self_flats += inf[3];
  		self_blocks += inf[4];
  	}
  	if (has_right[i]) {
  		vector<int> inf = surrounding_influencer(-1, i, player, state);
  		enemy_flats += inf[0];
  		enemy_blocks += inf[1];
  		enemy_caps += inf[2];
  		self_flats += inf[3];
  		self_blocks += inf[4];
  	}
  	if (has_up[i]) {
  		vector<int> inf = surrounding_influencer(-1*n, i, player, state);
  		enemy_flats += inf[0];
  		enemy_blocks += inf[1];
  		enemy_caps += inf[2];
  		self_flats += inf[3];
  		self_blocks += inf[4];
  	}
  	return {enemy_flats, enemy_blocks, enemy_caps, self_flats, self_blocks};
  }

  pair<double, double> evaluate_stack_strength(int i, vector<vector<pair<int, char>>> state) {
    double strength_player1 = 0;
    double strength_player2 = 0;
    int reserves = 0;
    int captives = 0;
    if (state[i].back().first == player) {
      //vector<int> influence = rough_influence_measure(i, this->player, state);
      //cerr << influence.size() << endl;
      for (int j = 0; j < state[i].size(); j++) {
        if (state[i][j].first == player) {
          reserves++;
        } else {
          captives++;
        }
      }
      /*
      int capstone = (state[i].back().second == 'C') ? 1 : 0;
      strength_player1 = (influence[3] + 1.5* influence[4]) + (1.5 * capstone * reserves) - 0.5 * state[i].size() * position_strength(i);
      strength_player1 -= (influence[0] * (1 - capstone)) + (1.5 * influence[1] * (captives / 2) * (1 - capstone)) + (2 * influence[2] * captives) + (captives * capstone);
      */

      if (state[i].back().second == 'C')
        strength_player1 += reserves + pow(captives, 1.6) - state[i].size() * position_strength(i);
      else if (state[i].back().second == 'S')
        strength_player1 += reserves + pow(captives, 1.4) - state[i].size() * position_strength(i);
      else
        strength_player1 += reserves + pow(captives, 1.2) - state[i].size() * position_strength(i);


    } else {
      //vector<int> influence = rough_influence_measure(i, 1 - this->player, state);
      //cerr << influence.size() << endl;
      for (int j = 0; j < state[i].size(); j++) {
        if (state[i][j].first == (1 - player)) {
          reserves++;
        } else {
          captives++;
        }
      }
      /*
      int capstone = (state[i].back().second == 'C') ? 1 : 0;
      strength_player2 = (influence[3] + 1.5 * influence[4]) + (1.5 * capstone * reserves) - 0.5 * state[i].size() * position_strength(i);
      strength_player2 -= (influence[0] * (1 - capstone)) + (1.5 * influence[1] * (captives / 2) * (1 - capstone)) + (2 * influence[2] * captives) + (captives * capstone);
      */

      if (state[i].back().second == 'C')
        strength_player2 += 1.5 * (pow(reserves, 1.6) + pow(captives, 2)) - 1.5 * state[i].size() * position_strength(i);
      else if (state[i].back().second == 'S')
        strength_player2 += reserves + pow(captives, 1.4) - 1.2 * state[i].size() * position_strength(i);
      else
        strength_player2 += reserves + pow(captives, 1.2) - state[i].size() * position_strength(i);

    }
    //cerr << "strength evaluated" << endl;
    return make_pair(strength_player1, strength_player2);
  }

  pair<bool, double> evaluation_function1(vector<vector<pair<int, char>>> state, vector<Player> players, int moves) {
    int size = n * n;
    int dim1 = 0;
  	int dim2 = 0;
  	int dimsum1 = 0;
  	int dimsum2 = 0;
  	vector<int> p1_isles;
  	vector<int> p2_isles;
    vector<bool> explored(n * n);
  	double strength_player1 = 0;
    double strength_player2 = 0;
    bool p1_rw = false;
    bool p2_rw = false;
    bool complete = true;
    int counter1 = 0;
    int counter2 = 0;

  	for (int i = 0; i < size; i++)
  	{
      if ((state[i].size() > 0)) {
        //cerr << "State Top is " << state[i].back().first << " ";

        /********  Evaluate Stack Strength  and Position Strength  *********/
        auto stack_evaluation = evaluate_stack_strength(i, state);
        strength_player1 += stack_evaluation.first;
        strength_player2 += stack_evaluation.second;

        if (!explored[i]) {
    			if ((state[i].back().first == player) && (state[i].back().second == 'F' || state[i].back().second == 'C'))
    			{
    				min_x = x[i];
    				max_x = x[i];
    				min_y = y[i];
    				max_y = y[i];
            int sum = flood_fill(i, this->player, state, &explored);
            counter1 += sum;
            //cerr << state[i].back().first << ", " << i << ", " << sum << " " << endl;
    				p1_isles.push_back(sum);
            int current_dim = 0;
            current_dim = std::max(max_y - min_y, max_x - min_x);
    				dim1 = std::max(current_dim, dim1);
    				dimsum1 = dimsum1 + dim1;
            strength_player1 += pow(sum, 1.2) + 2.5 * pow(current_dim, 2);
    			}
    			if ((state[i].back().first == (1 - player)) && (state[i].back().second == 'F' || state[i].back().second == 'C'))
    			{
    				min_x = x[i];
    				max_x = x[i];
    				min_y = y[i];
    				max_y = y[i];
    				int sum = flood_fill(i, 1 - this->player, state, &explored);
            counter2 += sum;
            //cerr << 1 - this->player << ", " << i << ", " << sum << " " << endl;
    				p2_isles.push_back(sum);
            int current_dim = 0;
            current_dim = std::max(max_y - min_y, max_x - min_x);
    				dim2 = std::max(current_dim, dim2);
    				dimsum2 = dimsum2 + dim2;
            strength_player2 += pow(sum, 2) + 3.5 * pow(current_dim, 2);
    			}
        }
      } else {
        complete = false;
      }
  	}
    //cerr << "Completed Exploration" << endl;
  	double eval1 = strength_player1 +  0.8 * players[this->player].flats + 3 * players[this->player].capstones;
  	double eval2 = strength_player2 +  0.8 * players[1 - this->player].flats + 3 * players[1 - this->player].capstones;

    //need to check flat wins and draw positions
    pair<bool, double> output;

    if (complete || !are_remaining_pieces(players)) {
      if (counter1 > counter2) {
        eval1 = 10000;
        eval2 = 0;
        output.first = true;
        output.second = eval1 - eval2;
      } else if (counter1 < counter2) {
        eval2 = 10000 ;
        eval1 = 0;
        output.first = true;
        output.second = eval1 - eval2;
      } else {
        if (players[this->player].flats > players[1 - this->player].flats) {
          eval1 = 10000;
          eval2 = 0;
          output.first = true;
          output.second = eval1 - eval2;
        } else if (players[this->player].flats < players[1 - this->player].flats) {
          eval2 = 10000;
          eval1 = 0;
          output.first = true;
          output.second = eval1 - eval2;
        } else {
          output.first = true;
          output.second = 6000;
        }
      }
    }

    //if (dim1 >= 3)
    //  cerr << "Max Length Path : " << dim1 << endl;
    if(dim1 == n - 1)
  	{
  		p1_rw = true;
  		eval1 = 20000;
  		eval2 = 0;
  	}
  	if(dim2 == n - 1)
  	{
  		p2_rw = true;
  		eval2 = 20000;
  		eval1 = 0;
  	}

    if (p1_rw || p2_rw) {
      //if (p1_rw)
      //  cerr << "Road Win possible!" << endl;
      output.first = true;
      output.second = eval1 - eval2;
    } else {
      output.first = false;
      output.second = eval1 - eval2;
    }
    return output;
  }

  pair<bool, double> evaluation_function2(vector<vector<pair<int, char>>> state, vector<Player> players, int moves) {
    return make_pair(false, features::diff_flats(state, this->player));
  }

  double minval(vector<vector<pair<int, char>>> state, vector<Player> players, double alpha, double beta, int depth,
                bool first, int moves) {
    //cerr << "Reached Evaluation Function" << endl;
    auto result = evaluation_function1(state, players, moves);
    //cerr << "Evaluation function computed" << endl;
    if (result.first || depth == 0) {
      return result.second;
    }
    double best_child = std::numeric_limits<double>::infinity();
    if (first) {
      vector<string> all_moves = this->game.generate_all_moves(this->player, true, state, players);
      for (auto iter = all_moves.begin(); iter != all_moves.end(); iter++) {
        vector<Player> current_players = players;
        auto current_state = this->game.execute_move(this->player, *iter, state, &current_players);
        double child = maxval(current_state, current_players, alpha, beta, depth - 1, false, moves + 1);
        beta = min(beta, child);
        if (alpha >= beta) {
          return child;
        }
        best_child = min(best_child, child);
      }

    } else {
      vector<string> all_moves = this->game.generate_all_moves(1 - this->player, false, state, players);
      for (auto iter = all_moves.begin(); iter != all_moves.end(); iter++) {
        vector<Player> current_players = players;
        auto current_state = this->game.execute_move(1 - this->player, *iter, state, &current_players);
        double child = maxval(current_state, current_players, alpha, beta, depth - 1, false, moves + 1);
        beta = min(beta, child);
        if (alpha >= beta) {
          return child;
        }
        best_child = min(best_child, child);
      }
    }
    return best_child;
  }

  double maxval(vector<vector<pair<int, char>>> state, vector<Player> players, double alpha, double beta, int depth,
                bool first, int moves) {
    //cerr << "Reached Evaluation Function" << endl;
    auto result = evaluation_function1(state, players, moves);
    //cerr << "Evaluation function computed" << endl;
    if (result.first || depth == 0) {
      return result.second;
    }
    double best_child = -1 * std::numeric_limits<double>::infinity();
    string best_move;
    if (first) {
      vector<string> all_moves = this->game.generate_all_moves(1 - this->player, true, state, players);
	  /*
      if (depth == depth_initial) {
        std::random_shuffle(all_moves.begin(), all_moves.end());
      }
      */
      for (auto iter = all_moves.begin(); iter != all_moves.end(); iter++) {
        vector<Player> current_players = players;
        auto current_state = this->game.execute_move(1 - this->player, *iter, state, &current_players);
        double child;
        if (this->player == 0) {
          child = minval(current_state, current_players, alpha, beta, depth - 1, true, moves + 1);
        } else {
          child = minval(current_state, current_players, alpha, beta, depth - 1, false, moves + 1);
        }
        alpha = max(alpha, child);
        if (alpha >= beta) {
          if (depth == depth_initial) {
            this->execute_optimal_move(*iter, first);
          }
          return child;
        }
        if (best_child < child) {
          best_child = child;
          best_move = *iter;
        }
      }
    } else {
      vector<string> all_moves = game.generate_all_moves(this->player, false, state, players);
      /*
      if (depth == depth_initial) {
        std::random_shuffle(all_moves.begin(), all_moves.end());
      }
      */
      for (auto iter = all_moves.begin(); iter != all_moves.end(); iter++) {
        vector<Player> current_players = players;
        auto current_state = this->game.execute_move(this->player, *iter, state, &current_players);
        double child = minval(current_state, current_players, alpha, beta, depth - 1, false, moves + 1);
        alpha = max(alpha, child);
        if (alpha >= beta) {
          if (depth == depth_initial) {
            this->execute_optimal_move(*iter, first);
          }
          return child;
        }
        if (best_child < child) {
          best_child = child;
          best_move = *iter;
        }
      }
    }
    if (depth == depth_initial) {
      this->execute_optimal_move(best_move, first);
    }
    return best_child;

  }

  void execute_optimal_move(string move, bool first) {
    //output optimal strategy and execute on the current state
    //cerr << "Chosen Move : " << move << endl;
    //cerr << "Executing Move" << endl;
    if (first) {
        this->state = this->game.execute_move(1 - this->player, move, this->state, &(this->players));
    } else {
      this->state = this->game.execute_move(this->player, move, this->state, &(this->players));
    }
    move = move + '\n';
    cout << move;
    gettimeofday(&finish, NULL);
    time_remaining = time_remaining - ((finish.tv_sec - start.tv_sec) + ((finish.tv_usec - start.tv_usec) / 1000000));
  }
};

int main() {
  srand(time(NULL));
  static_assert(std::numeric_limits<float>::is_iec559, "IEEE 754 required");
  minimax player;
  return 0;
}
