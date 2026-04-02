#ifndef game_c
#define game_c

void Game::initialize_game(int n) {
  this->n = n;
  this->total_squares = n * n;

  this->max_movable = n;
  this->max_up = n;
  this->max_down = 1;
  this->max_left = 'a';
  this->max_right = (char) ('a' + n - 1);

  for (int i = 0; i < this->total_squares; i++) {
    all_squares.push_back(this->square_to_string(i));
    //cerr << this->square_to_num(all_squares.back()) << endl;
  }
}

int Game::square_to_num(string square_string){
  //return -1 if the string is invalid
  if (square_string.size() != 2) {
    return -1;
  } else {
    int row, col;
    if (isalpha(square_string[0]) != 0) {
      if ((col = return_digit(square_string[1])) != -1) {
        row = (int) (square_string[0] - 96);
        if (row > this->n || col > this->n) {
          return -1;
        } else {
          return this->n * (col - 1) + (row - 1);
        }
      } else {
        return -1;
      }
    } else {
      return -1;
    }
  }
}

string Game::square_to_string(int square) {
  //convert square to string
  if (square > this->total_squares || square < 0) {
    return "";
  } else {
    string row = "";
    row.push_back((char)((square % this->n) + 97));
    string column = to_string((square / this->n) + 1);
    return row + column;
  }
}

vector<std::vector<int>> Game::partition(int n) {
  vector<std::vector<int>> output;
  vector<int> base_case;
  base_case.push_back(n);
  output.push_back(base_case);
  for (int i = 1; i < n; i++) {
    vector<vector<int>> small_partition = this->partition(n - i);
    auto iter = small_partition.begin();
    for (auto iter = small_partition.begin(); iter != small_partition.end(); iter++) {
      vector<int> temp;
      temp.push_back(i);
      temp.insert(temp.end(), (*iter).begin(), (*iter).end());
      output.push_back(temp);
    }
  }
  return output;
}

bool Game::check_valid(int square, char direction, vector<int> partition, vector<vector<pair<int, char>>> board) {
  //for checking valid movement, stack square can moved in the direction
  int change;
  int next_square;
  if (direction == '+') {
    change = this->n;
  } else if (direction == '-') {
    change = -1 * this->n;
  } else if (direction == '>') {
    change = 1;
  } else if (direction == '<') {
    change = -1;
  }
  for (int i = 0; i < partition.size(); i++) {
    next_square = square + change * (i + 1);
    int length = board[next_square].size();
    if (length > 0 && board[next_square].back().second == 'C')
			return false;
		if (length > 0 && board[next_square].back().second == 'S' && (i != (partition.size() - 1)))
			return false;
		if (i == (partition.size() - 1) && length > 0 && board[next_square].back().second == 'S' && partition[i] > 1)
			return false;
		if (i == (partition.size() - 1) && length > 0 && board[next_square].back().second == 'S' && board[square].back().second != 'C')
			return false;
  }
  return true;
}

vector<string> Game::generate_stack_moves(int square, vector<vector<pair<int, char>>> board) {
  //generate stack moves from square, assumes active player is topmost color

  vector<string> all_moves;
  int r = square % this->n;
  int c = square / this->n;
  int size = board[square].size();
  char dirs[4] = {'+', '-', '<', '>'};
  int up = this->n - 1 - c;
  int down = c;
  int right = this->n - 1 - r;
  int left = r;
  int rem_squares[4] = {up, down, left, right};
  for (int i = 0; i < min(size, this->n); i++) {
    auto part_list = this->partition(i + 1);
    for (int di = 0; di < 4; di++) {
      for (auto iter = part_list.begin(); iter != part_list.end(); iter++) {
        if ((*iter).size() <= rem_squares[di]) {
          if (this->check_valid(square, dirs[di], (*iter), board)) {
            string move = "";
            int sum = 0;
            for (auto parts = (*iter).begin(); parts != (*iter).end(); parts++) {
              move = move + to_string(*parts);
              sum += (*parts);
            }
            move = to_string(sum) + this->all_squares[square] + dirs[di] + move;
            all_moves.push_back(move);
          }
        }
      }
    }
  }
  return all_moves;
}

vector<string> Game::generate_all_moves(int player, bool first, vector<vector<pair<int, char>>> board,
                                        vector<Player> players) {
  //generate all possible moves for player, returns a list of move strings

	vector<string> all_moves;
  for (int i = 0; i < this->total_squares; i++) {
    if (board[i].size() == 0) {
      if (players[player].flats > 0) {
        string move = "F" + this->all_squares[i];
        all_moves.push_back(move);
      }
      //cannot move standing stone of other player
      if (!first && players[player].flats > 0) {
        string move = "S" + this->all_squares[i];
        all_moves.push_back(move);
      }
      //cannot move capstone of other player
      if (!first && players[player].capstones > 0) {
        string move = "C" + this->all_squares[i];
        all_moves.push_back(move);
      }
    }
  }

  for (int i = 0; i < this->total_squares; i++) {
    if (board[i].size() > 0 && board[i].back().first == player && !first) {
      auto stack_moves = this->generate_stack_moves(i, board);
      all_moves.insert(all_moves.end(), stack_moves.begin(), stack_moves.end());
    }
  }
  return all_moves;
}

vector<vector<pair<int, char>>> Game::execute_move(int current_piece, string move_string,
                                                  vector<vector<pair<int, char>>> board, vector<Player> *players) {
  //execute move on the board
  //cerr << isalpha(move_string[0]) << " " << return_digit(move_string[0]) << endl;
  if (isalpha(move_string[0]) != 0) {
		int square = this->square_to_num(move_string.substr(1, move_string.size() - 1));
		if (move_string[0] == 'F' || move_string[0] == 'S') {
			board[square].push_back(make_pair(current_piece, move_string[0]));
      //print_vector(board[square]);
			(*players)[current_piece].flats -= 1;
    }
		else if (move_string[0] == 'C') {
			board[square].push_back(make_pair(current_piece, move_string[0]));
      //print_vector(board[square]);
			(*players)[current_piece].capstones -= 1;
    }
  }
	else if (return_digit(move_string[0]) != -1) {
		int count = (int)(move_string[0] - '0');
    //cerr << "count is : " << count << endl;
		int square = this->square_to_num(move_string.substr(1, 2));
		char direction = move_string[3];
    int change;
		if (direction == '+')
			change = this->n;
		else if (direction == '-')
			change = -1 * this->n;
		else if (direction == '>')
			change = 1;
		else if (direction == '<')
			change = -1;

    int prev_square = square;
    for (int i = 4; i < move_string.size(); i++) {
      int next_count = (int)(move_string[i] - '0');
      //cerr << "next count is : " << next_count << endl;
      int next_square = prev_square + change;
      if (board[next_square].size() > 0 && board[next_square].back().second == 'S') {
        board[next_square][board[next_square].size() - 1] = make_pair(board[next_square].back().first, 'F');
      }
      if (next_count == count) {
        auto begin = board[square].begin() + (board[square].size() - count);
        auto end = board[square].end();
        board[next_square].insert(board[next_square].end(), begin, end);
        //print_vector(board[next_square]);
      } else {
        auto begin = board[square].begin() + (board[square].size() - count);
        auto end = board[square].begin() + (board[square].size() - count + next_count);
        board[next_square].insert(board[next_square].end(), begin, end);
        //print_vector(board[next_square]);
      }
      prev_square = next_square;
      count -= next_count;
    }

		count = (int)(move_string[0] - '0');
    auto begin = board[square].begin() + (board[square].size() - count);
    auto end = board[square].end();
		board[square].erase(begin, end);
    //print_vector(board[square]);
  }
  return board;
}
#endif
