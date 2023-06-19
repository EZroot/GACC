pub struct WindowInfo{
    title: String,
    size: (i32, i32),
    resizable: bool,
    force_dark_theme: bool,
}

impl WindowInfo{
    pub fn new(        
        title: String,
        win_size: (i32, i32),
        resizable: bool,
        force_dark_theme: bool,) -> Self{
        Self {
            title: title,
            size: win_size,
            resizable: resizable,
            force_dark_theme: force_dark_theme,
        }
    }

    pub fn clone(&self) -> Self {
        Self {
            title: self.title.clone(),
            size: self.size,
            resizable: self.resizable,
            force_dark_theme: self.force_dark_theme,
        }
    }

    pub fn get_title(&self) -> String {
        self.title.clone()
    }

    pub fn get_win_size(&self) -> (i32,i32) {
        self.size
    }

    pub fn get_resizable(&self) -> bool {
        self.resizable
    }

    pub fn get_force_dark_theme(&self) -> bool {
        self.force_dark_theme
    }
}