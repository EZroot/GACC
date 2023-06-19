use controller::winbuilder;
use model::windowinfo::WindowInfo;
use model::windowapp::WindowsApp;
use std::cell::RefCell;
use std::rc::Rc;

use chrono::Local;
use glib::*;
use tokio::*;
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

mod controller {
    pub mod winbuilder;
}
mod model {
    pub mod airequestor;
    pub mod appstate;
    pub mod gestures;
    pub mod windowinfo;
    pub mod windowapp;
}
mod view {
    pub mod window;
}

fn main() {
    let window_info = WindowInfo::new("EZPic".to_string(), (800,600), true, true);
    let _ = winbuilder::WindowBuilder::new(window_info);
}
