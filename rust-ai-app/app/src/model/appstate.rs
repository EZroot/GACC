pub struct AppState {
    pub image_surface: Option<gtk::cairo::ImageSurface>,
    pub left_drawing_surface: Option<gtk::cairo::Surface>,
    pub right_drawing_surface: Option<gtk::cairo::Surface>,
    pub start_pos: (f64, f64),
    pub prompt: String,
    pub image_size: (i32,i32),
    pub image_count: i32,
    pub load_image_as_mask: bool,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            image_surface: None,
            left_drawing_surface: None,
            right_drawing_surface: None,
            start_pos: (0.0, 0.0),
            prompt: "Example Prompt".to_string(),
            image_size: (512,768),
            image_count: 1,
            load_image_as_mask: true,
        }
    }
}