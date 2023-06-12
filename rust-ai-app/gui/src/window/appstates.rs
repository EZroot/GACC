pub struct AppState {
    pub surface: Option<gtk::cairo::Surface>,
    pub start_pos: (f64, f64),
    pub prompt: String,
    pub image_size: (i32,i32),
    pub image_count: i32
}

impl AppState {
    pub fn new() -> Self {
        Self {
            surface: None,
            start_pos: (0.0, 0.0),
            prompt: "Example Prompt".to_string(),
            image_size: (512,768),
            image_count: 1
        }
    }
}