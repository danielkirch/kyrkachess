use crate::board::Board;
use crate::game_history::GameHistory;

#[derive(Clone, Copy)]
pub struct Game {
    pub board: Board,
    pub history: GameHistory,
}

impl Game {
    pub fn new() -> Game {
        Game {
            board: Board::starting_position(),
            history: GameHistory::new(),
        }
    }

    pub fn from_fen(fen: &str) -> Game {
        Game {
            board: Board::from_fen(fen).unwrap(),
            history: GameHistory::from_fen(fen),
        }
    }

    pub fn make_move(&mut self, mv: u16) {
        let captured_piece = self.board.make_move(mv);
        self.history.record_move(mv, captured_piece);
    }

    pub fn undo_move(&mut self) {
        if let Some(state) = self.history.undo_move() {
            self.board
                .undo_move(state.current_move.unwrap(), state.captured_piece);
        }
    }

    pub fn generate_legal_moves(&self) -> Vec<u16> {
        let white_to_move = self.history.current_state().white_to_move;
        let moves = self.board.generate_legal_moves(white_to_move);
        let mut legal_moves: Vec<u16> = Vec::new();

        // filter out castling and en passant that are not legal due to game state
        let state = self.history.current_state();
        for mv in moves.into_iter() {
            if mv & 0xF000 == 0x5000 {
                let to = ((mv >> 6) & 0x3F) as u8;
                // en passant
                if state.en_passant_square != Some(to) {
                    continue;
                }
            } else if mv & 0xE000 == 0x2000 {
                // castling
                if mv & 0xF000 == 0x2000 {
                    // kingside
                    if state.white_to_move {
                        if state.castling_rights & 0x01 == 0 {
                            continue;
                        }
                    } else {
                        if state.castling_rights & 0x04 == 0 {
                            continue;
                        }
                    }
                } else if mv & 0xF000 == 0x3000 {
                    // queenside
                    if state.white_to_move {
                        if state.castling_rights & 0x02 == 0 {
                            continue;
                        }
                    } else {
                        if state.castling_rights & 0x08 == 0 {
                            continue;
                        }
                    }
                }
            }
            legal_moves.push(mv);
        }

        legal_moves
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::Piece;

    #[test]
    fn test_starting_position() {
        let game = Game::new();
        let board = game.board;
        assert_eq!(board.piece_list[0], Piece::WhiteRook);
        assert_eq!(board.piece_list[60], Piece::BlackKing);
    }

    #[test]
    fn test_from_fen() {
        // Test parsing starting position FEN
        let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        let game = Game::from_fen(fen);
        let board = game.board;
        assert_eq!(board.piece_list[0], Piece::WhiteRook);
        assert_eq!(board.piece_list[60], Piece::BlackKing);
    }

    #[test]
    fn test_from_fen_e4e5_opening() {
        // Test parsing a FEN string after 1. e4 e5
        let fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2";
        let game = Game::from_fen(fen);
        let board = game.board;
        assert_eq!(board.piece_list[28], Piece::WhitePawn); // e4
        assert_eq!(board.piece_list[36], Piece::BlackPawn); // e5
    }

    #[test]
    fn test_from_fen_bongcloud_opening() {
        // Test parsing a FEN string after 1. e4 e5 2. Ke2
        let fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPPKPPP/RNBQ1BNR b kq - 1 2";
        let game = Game::from_fen(fen);
        let board = game.board;
        assert_eq!(board.piece_list[28], Piece::WhitePawn); // e4
        assert_eq!(board.piece_list[36], Piece::BlackPawn); // e5
        assert_eq!(board.piece_list[12], Piece::WhiteKing); // e2
    }
}
