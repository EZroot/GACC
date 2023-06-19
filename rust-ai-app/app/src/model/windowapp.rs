use std::cell::RefCell;
use std::rc::Rc;

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

use crate::model::appstate::AppState;
use crate::model::windowinfo;

use super::windowinfo::WindowInfo;
pub struct WindowsApp {
    window_info: windowinfo::WindowInfo,
    gtk_window: gtk::ApplicationWindow,
    app_state: Rc<RefCell<AppState>>,
}

impl WindowsApp {
    pub fn new(
        application: &gtk::Application,
        window_info: windowinfo::WindowInfo,
    ) -> Self {
        let app_state = Rc::new(RefCell::new(AppState::new()));
        let gtk_window = Self::create_new_window(application, window_info.clone());
        Self {
            window_info: window_info,
            gtk_window: gtk_window,
            app_state: app_state,
        }
    }

    pub fn get_gtk_window(&self) -> gtk::ApplicationWindow{
        self.gtk_window.clone()
    }

    fn create_new_window(
        application: &gtk::Application,
        window_info: windowinfo::WindowInfo,
    ) -> gtk::ApplicationWindow {

        let window = gtk::ApplicationWindow::new(application);
        window.set_title(Some(&window_info.get_title()));
        window
            .settings()
            .set_gtk_application_prefer_dark_theme(window_info.get_force_dark_theme());
        let win_size = window_info.get_win_size();
        window.set_default_size(win_size.0, win_size.1);
        window.set_resizable(window_info.get_resizable());
        window
    }
}

