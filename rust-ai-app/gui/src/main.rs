pub mod window{
    pub mod appstates;
    pub mod wingui;
    
    pub mod components {
        pub mod airequestor;
        pub mod gestures;
        pub mod helper;
    }
}

use gtk::{Application, prelude::*, gio::ApplicationFlags};
use window::wingui::WindowsApp;
use tokio::runtime::Builder;

fn main() {
    let application = Application::builder()
        .application_id("com.github.gtk-rs.examples.basic")
        .flags(ApplicationFlags::empty())
        .build();

    let windows_gui_app = WindowsApp::new("Picasso".to_string(), (1280,768), true, true);

    application.connect_activate(move |app| {
        // Create a Tokio runtime
        let rt = Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        // Spawn the async function on the Tokio runtime
        rt.block_on(async {
            windows_gui_app.on_activate(app).await;
        });
    });

    application.run();
}
