use crate::model::windowapp::WindowsApp;
use crate::model::windowinfo::WindowInfo;

use chrono::Local;
use glib::*;

use gtk::cairo::ffi::cairo_font_type_t;
use gtk::cairo::{self, Context, Surface};
use gtk::ffi::GtkResponseType;
use gtk::gdk::ffi::{GdkRGBA, GDK_BUTTON_PRIMARY};
use gtk::gio::{self, ApplicationFlags};
// glib and other dependencies are re-exported by the gtk crate
use gtk::gdk::{Display, Paintable};
use gtk::gdk_pixbuf::{self, Pixbuf};
use gtk::graphene::ffi::{graphene_rect_init, graphene_rect_t};
use gtk::{
    gdk, glib, graphene, gsk, CssProvider, FileChooser, Inhibit, ResponseType, StateFlags,
    STYLE_PROVIDER_PRIORITY_APPLICATION,
};
use gtk::{prelude::*, Application};
use gtk::{prelude::*, ProgressBar};
use serde::{Deserialize, Serialize};
use tokio::runtime::Builder;

pub struct WindowBuilder {
}
impl WindowBuilder {
    pub fn new(window_info: WindowInfo) {
        let application = Application::builder()
            .application_id("com.github.gtk-rs.examples.basic")
            .flags(ApplicationFlags::empty())
            .build();

        application.connect_activate( move |app| {
            // Create a Tokio runtime
            let window_info = window_info.clone();
            let windowsapp: WindowsApp = WindowsApp::new(app, window_info);
            // Spawn the async function on the Tokio runtime
            let window = windowsapp.get_gtk_window();
            window.present();
            // window.on_activate(app).await;
        });

        application.run();
    }

}
