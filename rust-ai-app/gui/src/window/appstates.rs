pub struct AppState {
    pub surface: Option<gtk::cairo::Surface>,
    pub start_pos: (f64, f64),
}

impl AppState {
    pub fn new() -> Self {
        Self {
            surface: None,
            start_pos: (0.0, 0.0),
        }
    }
}