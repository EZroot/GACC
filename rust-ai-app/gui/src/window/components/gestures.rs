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

pub struct Gestures {}

impl Gestures {
    pub fn clear_surface(app_state: &AppState) {
        if let Some(surface) = &app_state.surface {
            let cr = Context::new(surface).unwrap();
            cr.set_source_rgb(1.0, 1.0, 1.0);
            cr.paint().unwrap();
        }
    }

    pub fn draw_brush(
        drawing_area: &gtk::DrawingArea,
        x: f64,
        y: f64,
        app_state: &Rc<RefCell<AppState>>,
    ) {
        if let Some(surface) = &app_state.borrow().surface {
            let cr = Context::new(surface).unwrap();
            cr.rectangle(x - 2.0, y - 2.0, 4.0, 4.0);
            cr.set_source_rgba(1.0, 0.0, 0.0, 1.0);
            cr.fill().unwrap();
            drawing_area.queue_draw();
        }
    }

    pub fn drag_begin(
        x: f64,
        y: f64,
        drawing_area: &gtk::DrawingArea,
        app_state: &Rc<RefCell<AppState>>,
    ) {
        app_state.borrow_mut().start_pos = (x, y);
        //draw_brush(drawing_area, x, y, app_state);
    }

    pub fn drag_update(
        x: f64,
        y: f64,
        drawing_area: &gtk::DrawingArea,
        app_state: &Rc<RefCell<AppState>>,
    ) {
        let (start_x, start_y) = app_state.borrow().start_pos;
        Gestures::draw_brush(drawing_area, start_x + x, start_y + y, app_state);
    }

    pub fn pressed(
        x: f64,
        y: f64,
        drawing_area: &gtk::DrawingArea,
        app_state: &Rc<RefCell<AppState>>,
    ) {
        Gestures::clear_surface(&app_state.borrow());
        drawing_area.queue_draw();
    }
}
