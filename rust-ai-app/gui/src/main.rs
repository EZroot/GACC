pub mod window{
    pub mod appstates;
    pub mod wingui;
    
pub mod components {
    pub mod gestures;
    pub mod helper;
    }
}


use window::wingui;
use window::appstates;

use window::wingui::WindowsApp;
use gtk::{gio::ApplicationFlags, prelude::{ApplicationExt, ApplicationExtManual}};
use window::components;

fn main() {
    // Create a new application with the builder pattern
    let app = gtk::Application::builder()
        .application_id("com.github.gtk-rs.examples.basic")
        .flags(ApplicationFlags::FLAGS_NONE)
        .build();

    let windowGuiApp = WindowsApp::new("Pico Picasso".to_string(), (512,768));

    app.connect_activate(move |x| {windowGuiApp.on_activate(x);});

    // Run the application
    app.run();
}