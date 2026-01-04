use crate::constants::Piece;
use crate::game::Game;
use crate::zobrist::Zobrist;
use rand::seq::IndexedRandom;
use std::io;

pub struct UciInterface {
    game: Option<Game>,
    zobrist: Zobrist,
}

impl UciInterface {
    pub fn new() -> Self {
        UciInterface {
            game: None,
            zobrist: Zobrist::new(),
        }
    }

    pub fn run(&mut self) {
        let mut input = String::new();
        loop {
            input.clear();
            io::stdin().read_line(&mut input).unwrap();
            let command = input.trim();
            match command {
                "quit" => {
                    break;
                }
                "uci" => {
                    println!("id name rusty_chess");
                    println!("id author Daniel Kirch");
                    println!("uciok");
                }
                "isready" => {
                    println!("readyok");
                }
                pos_str if pos_str.starts_with("position") => {
                    // Handle position command
                    self.parse_position_command(pos_str);
                }
                go_str if go_str.starts_with("go") => {
                    // Handle go command
                    self.make_move();
                }
                _ => {
                    println!("Unknown command: {}", command);
                }
            }
        }
    }

    fn parse_position_command(&mut self, command: &str) {
        // This parses the position command. If successful, it sets up the game state.
        // Example: position startpos moves e2e4 e7e5
        if !command.starts_with("position") {
            return;
        }

        let parts: Vec<&str> = command[8..].split("moves").collect();
        // trim all parts
        let parts: Vec<&str> = parts.iter().map(|s| s.trim()).collect();

        // Set up initial position
        if parts[0] == "startpos" {
            // Set up starting position
            self.game = Some(Game::new());
        } else if parts[0].starts_with("fen") {
            // Set up position from FEN
            let fen = parts[0][4..].trim();
            self.game = Some(Game::from_fen(fen));
        }

        // Apply moves if any
        if parts.len() > 1 {
            let moves: Vec<&str> = parts[1].split_whitespace().collect();
            for mv_str in moves {
                let mv = self.parse_move_string(mv_str);
                if let Some(mv) = mv {
                    let mut game: Game = self.game.unwrap();
                    game.make_move(mv);
                    self.game = Some(game);
                } else {
                    println!("info invalid move: {}", mv_str);
                }
            }
        }

        println!("info position set up");
        println!(
            "info current board position: {:?}",
            self.game.unwrap().board.piece_list
        );
    }

    fn parse_move_string(&self, mv_str: &str) -> Option<u16> {
        let board = self.game.unwrap().board;

        // Convert move string like "e2e4" to move bitmap u16
        if mv_str.len() < 4 {
            return None;
        }
        let from_file = mv_str.chars().nth(0).unwrap() as u8 - 'a' as u8;
        let from_rank = mv_str.chars().nth(1).unwrap() as u8 - '1' as u8;
        let to_file = mv_str.chars().nth(2).unwrap() as u8 - 'a' as u8;
        let to_rank = mv_str.chars().nth(3).unwrap() as u8 - '1' as u8;

        let from = (from_rank * 8 + from_file) as usize;
        let to = (to_rank * 8 + to_file) as usize;

        // Check for promotion
        let promotion = if mv_str.len() == 5 {
            match mv_str.chars().nth(4).unwrap() {
                'q' => 0x8000,
                'r' => 0x9000,
                'b' => 0xA000,
                'n' => 0xB000,
                _ => 0x0000,
            }
        } else {
            0x0000
        };

        // Check for capture
        let capture = if board.piece_list[to] != Piece::Empty {
            0x4000
        } else {
            0x0000
        };

        // Check for en passant
        let game_state = self.game.unwrap().history.current_state();
        let is_pawn = match board.piece_list[from] {
            Piece::WhitePawn | Piece::BlackPawn => true,
            _ => false,
        };
        let en_passant = if to as u8 == game_state.en_passant_square.unwrap_or(64) && is_pawn {
            0x5000
        } else {
            0x0000
        };

        // Check for double pawn push
        let double_pawn_push = if is_pawn && ((from as i8 - to as i8).abs() == 16) {
            0x1000
        } else {
            0x0000
        };

        // Check for castling
        let is_king = match board.piece_list[from] {
            Piece::WhiteKing | Piece::BlackKing => true,
            _ => false,
        };
        let castle = if is_king {
            match from as i8 - to as i8 {
                -2 => 0x2000, // kingside
                2 => 0x3000,  // queenside
                _ => 0x0000,
            }
        } else {
            0x0000
        };

        // Construct move bitmap
        let mut mv: u16 = 0;
        mv |= from as u16;
        mv |= (to as u16) << 6;
        mv |= promotion;
        mv |= capture;
        mv |= en_passant;
        mv |= double_pawn_push;
        mv |= castle;

        Some(mv)
    }

    fn make_move(&mut self) {
        let game: Game = self.game.unwrap();
        // For demonstration, we will make a random valid move.
        let moves = game.generate_legal_moves();
        match moves.choose(&mut rand::rng()) {
            Some(mv) => println!("bestmove {}", UciInterface::move_to_string(*mv)),
            None => println!("bestmove (none)"),
        }
    }

    fn move_to_string(mv: u16) -> String {
        let from = (mv & 0x3F) as u8;
        let to = ((mv >> 6) & 0x3F) as u8;

        let from_file = (from % 8) + b'a';
        let from_rank = (from / 8) + b'1';
        let to_file = (to % 8) + b'a';
        let to_rank = (to / 8) + b'1';

        let mut move_str = String::new();
        move_str.push(from_file as char);
        move_str.push(from_rank as char);
        move_str.push(to_file as char);
        move_str.push(to_rank as char);

        // Check for promotion
        let promotion = mv & 0x8000;
        if promotion != 0 {
            let promo_char = match promotion {
                0x8000 => 'q',
                0x9000 => 'r',
                0xA000 => 'b',
                0xB000 => 'n',
                _ => ' ',
            };
            move_str.push(promo_char);
        }

        move_str
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_move_string() {
        let mut interface = UciInterface::new();
        let command = "position startpos moves e2e4 e7e5";
        interface.parse_position_command(command);
        let board = interface.game.unwrap().board;
        assert_eq!(board.piece_list[28], Piece::WhitePawn);
        assert_eq!(board.piece_list[36], Piece::BlackPawn);
    }

    #[test]
    fn test_move_to_string() {
        let mv = 0x000C | (0x001C << 6);
        let move_str = UciInterface::move_to_string(mv);
        assert_eq!(move_str, "e2e4");
    }

    #[test]
    fn test_move_to_string_with_promotion() {
        let mv = 0x0036 | (0x003E << 6) | 0x8000;
        let move_str = UciInterface::move_to_string(mv);
        assert_eq!(move_str, "g7g8q");
    }
}
