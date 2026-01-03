use crate::constants::MAX_GAME_HISTORY_LENGTH;
use crate::constants::Piece;
use crate::game_state::GameState;

#[derive(Clone, Copy)]
pub struct GameHistory {
    pub list: [GameState; MAX_GAME_HISTORY_LENGTH],
    pub length: usize,
}

impl GameHistory {
    pub fn new() -> GameHistory {
        GameHistory {
            list: [GameState::new(); MAX_GAME_HISTORY_LENGTH],
            length: 1,
        }
    }

    pub fn from_fen(fen: &str) -> GameHistory {
        let mut list = [GameState::new(); MAX_GAME_HISTORY_LENGTH];
        list[0] = GameState::from_fen(fen).unwrap();
        GameHistory {
            list: list,
            length: 1,
        }
    }

    pub fn current_state(&self) -> GameState {
        self.list[self.length - 1]
    }

    pub fn record_move(&mut self, mv: u16, captured_piece: Piece) -> Option<GameState> {
        if self.length < MAX_GAME_HISTORY_LENGTH {
            let prev_state = self.list[self.length - 1];
            let new_state = prev_state.after_move(mv, captured_piece);
            self.list[self.length] = new_state;
            self.length += 1;
            Some(new_state)
        } else {
            None
        }
    }

    pub fn undo_move(&mut self) -> Option<GameState> {
        if self.length > 1 {
            let state = self.list[self.length - 1];
            self.length -= 1;
            Some(state)
        } else {
            None
        }
    }
}
