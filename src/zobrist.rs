use crate::game::Game;
use rand::prelude::*;

pub struct Zobrist {
    // store Zobrist hash values
    piece_table: [[u64; 64]; 13],
    castling_rights: [u64; 4],
    en_passant_square: [u64; 64],
    side_to_move: u64,
}

impl Zobrist {
    pub fn new() -> Self {
        let mut rng = StdRng::seed_from_u64(0xDEADBEEF); // fixed seed for reproducibility
        let mut piece_table = [[0u64; 64]; 13];
        for piece in 0..12 {
            for square in 0..64 {
                piece_table[piece][square] = rng.random::<u64>();
            }
        }
        let mut castling_rights = [0u64; 4];
        for i in 0..4 {
            castling_rights[i] = rng.random::<u64>();
        }
        let mut en_passant_square = [0u64; 64];
        for i in 0..16 {
            en_passant_square[i] = rng.random::<u64>();
        }
        let side_to_move = rng.random::<u64>();
        Zobrist {
            piece_table,
            castling_rights,
            en_passant_square,
            side_to_move,
        }
    }

    pub fn zobrist_hash(&self, game: Game) -> u64 {
        let mut hash = 0u64;
        // Pieces on squares
        for square in 0..64 {
            let piece = game.board.piece_list[square] as usize;
            hash ^= self.piece_table[piece][square];
        }
        // get latest game state
        let state = game.history.current_state();
        // Castling rights
        for i in 0..4 {
            if (state.castling_rights & (1 << i)) != 0 {
                hash ^= self.castling_rights[i];
            }
        }
        // En passant square
        if state.en_passant_square.is_some() {
            let file = state.en_passant_square.unwrap() % 8;
            hash ^= self.en_passant_square[file as usize];
        }
        // Side to move
        if state.white_to_move {
            hash ^= self.side_to_move;
        }
        hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zobrist_hash() {
        let zobrist = Zobrist::new();
        let game = Game::new();
        let hash = zobrist.zobrist_hash(game);
        assert_ne!(hash, 0);
    }
}
