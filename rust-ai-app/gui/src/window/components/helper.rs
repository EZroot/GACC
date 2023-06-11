use std::cell::RefCell;
use std::rc::Rc;

use chrono::Local;
use glib::*;

use gtk::cairo::ffi::cairo_font_type_t;
use gtk::cairo::{self, Context};
use gtk::gdk::ffi::{GdkRGBA, GDK_BUTTON_PRIMARY};
use gtk::gio::ApplicationFlags;
// glib and other dependencies are re-exported by the gtk crate
use gtk::gdk::{Display, Paintable};
use gtk::graphene::ffi::{graphene_rect_init, graphene_rect_t};
use gtk::{gdk, glib, graphene, gsk, CssProvider, STYLE_PROVIDER_PRIORITY_APPLICATION};
use gtk::{prelude::*, ProgressBar};
use gtk::gdk_pixbuf::Pixbuf;

use crate::window::appstates::AppState;

pub struct Helper {}

impl Helper {
    pub fn current_time() -> String {
        format!("{}", Local::now().format("%Y-%m-%d %H:%M:%S"))
    }
    
    pub fn print_test() {
        println!("Button Test");
    }
}
