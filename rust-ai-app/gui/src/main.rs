mod custom_paintable;

use std::cell::RefCell;
use std::rc::Rc;

use custom_paintable::CustomPaintable;
use glib::*;

use gtk::cairo::{Context, self};
use gtk::cairo::ffi::cairo_font_type_t;
use gtk::gdk::ffi::GdkRGBA;
use gtk::gio::ApplicationFlags;
// glib and other dependencies are re-exported by the gtk crate
use gtk::{glib, graphene, gdk, gsk};
use gtk::graphene::ffi::{graphene_rect_t, graphene_rect_init};
use gtk::prelude::*;
use gtk::gdk::Paintable;

struct AppState {
    surface: Option<gtk::cairo::Surface>,
    start_pos: (f64, f64),
}

impl AppState {
    fn new() -> Self {
        Self {
            surface: None,
            start_pos: (0.0, 0.0),
        }
    }
}

fn clear_surface(app_state: &AppState) {
    if let Some(surface) = &app_state.surface {
        let cr = Context::new(surface).unwrap();
        cr.set_source_rgb(1.0, 1.0, 1.0);
        cr.paint();
    }
}

fn draw_cb(drawing_area: &gtk::DrawingArea, cr: &gtk::cairo::Context, width: i32, height: i32, app_state: &Rc<RefCell<AppState>>) {
    if let Some(surface) = &app_state.borrow().surface {
        cr.set_source_surface(surface, 0.0, 0.0).unwrap();
        cr.paint().unwrap();
    }
}

fn draw_brush(drawing_area: &gtk::DrawingArea, x: f64, y: f64, app_state: &Rc<RefCell<AppState>>) {
    if let Some(surface) = &app_state.borrow().surface {
        let cr = Context::new(surface).unwrap();
        cr.rectangle(x - 3.0, y - 3.0, 6.0, 6.0);
        cr.fill().unwrap();
    }
    drawing_area.queue_draw();
}

fn drag_begin(x: f64, y: f64, drawing_area: &gtk::DrawingArea, app_state: &Rc<RefCell<AppState>>) {
    app_state.borrow_mut().start_pos = (x, y);
    draw_brush(drawing_area, x, y, app_state);
}

fn drag_update(x: f64, y: f64, drawing_area: &gtk::DrawingArea, app_state: &Rc<RefCell<AppState>>) {
    let (start_x, start_y) = app_state.borrow().start_pos;
    draw_brush(drawing_area, start_x + x, start_y + y, app_state);
}

fn pressed(x: f64, y: f64, drawing_area: &gtk::DrawingArea, app_state: &Rc<RefCell<AppState>>) {
    clear_surface(&app_state.borrow());
    drawing_area.queue_draw();
}

// When the application is launched…
fn on_activate(application: &gtk::Application) {
    // … create a new window …
    let window = gtk::ApplicationWindow::new(application);
    window.set_title(Some("Pico Picasso"));
    // … with a button in it …
    let button = gtk::Button::with_label("Hello World!");
    // … which closes the window when clicked
    button.connect_clicked(clone!(@weak window => move |_| print_test()));

    // let paintable = CustomPaintable::new();
    // let picture = gtk::Picture::new();
    // picture.set_halign(gtk::Align::Center);
    // picture.set_size_request(800, 600);
    // picture.set_paintable(Some(&paintable));
    


    let drawing_area = gtk::DrawingArea::new();
    drawing_area.set_size_request(512, 768);

    
    let motion_controller = gtk::EventControllerMotion::new();
    motion_controller.connect_motion(clone!(@weak drawing_area => move |controller,x,y| {on_motion_event(controller, x , y, &drawing_area);
    }));


    drawing_area.add_controller(motion_controller);
    
    let image_surface = gdk::cairo::ImageSurface::create(cairo::Format::ARgb32, 512, 768).unwrap();
    let surface = &*image_surface;

    drawing_area.set_draw_func(clone!(@strong surface => move |drawing_area, cr, width, height| {
            cr.set_source_surface(&surface, 512.0, 768.0).unwrap();
            cr.paint().unwrap();
    }));

    //picture.add_controller(motion_controller);


    let frame = gtk::Frame::new(None);
    window.set_child(Some(&frame));

    let win_box = gtk::Box::new(gtk::Orientation::Vertical, 12);
    win_box.append(&button);
    win_box.append(&drawing_area);
    window.set_child(Some(&win_box));
    window.present();
}

fn on_motion_event(controller: &gtk::EventControllerMotion, x:f64, y:f64, drawing : &gtk::DrawingArea) {
    println!("Motion detected x{}/y{}",x,y);
    
    //drawing.set_draw_pixel(x,y);
    //drawing.snapshot(snapshot, width, height)
}

fn print_test(){
    println!("Button Test");
}

fn main() {
    // Create a new application with the builder pattern
    let app = gtk::Application::builder()
        .application_id("com.github.gtk-rs.examples.basic")
        .flags(ApplicationFlags::FLAGS_NONE)
        .build();

    app.connect_activate(on_activate);
    
    // Run the application
    app.run();
}
