mod custom_paintable;

use std::cell::RefCell;
use std::rc::Rc;

use custom_paintable::CustomPaintable;
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
        cr.paint().unwrap();
    }
}

fn draw_brush(drawing_area: &gtk::DrawingArea, x: f64, y: f64, app_state: &Rc<RefCell<AppState>>) {
    if let Some(surface) = &app_state.borrow().surface {
        let cr = Context::new(surface).unwrap();
        cr.rectangle(x - 2.0, y - 2.0, 4.0, 4.0);
        cr.set_source_rgba(1.0, 0.0, 0.0, 1.0);
        cr.fill().unwrap();
        drawing_area.queue_draw();
    }
}

fn drag_begin(x: f64, y: f64, drawing_area: &gtk::DrawingArea, app_state: &Rc<RefCell<AppState>>) {
    app_state.borrow_mut().start_pos = (x, y);
    //draw_brush(drawing_area, x, y, app_state);
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
    window
        .settings()
        .set_gtk_application_prefer_dark_theme(true);
    window.set_resizable(false);

    let app_state = Rc::new(RefCell::new(AppState::new()));

    let frame = gtk::Frame::new(None);

    // -------------------------------------------

    let header_bar = gtk::HeaderBar::new();
    window.set_titlebar(Some(&header_bar));

    let label = gtk::Label::builder()
        .label("Type to start search")
        .halign(gtk::Align::Center)
        .valign(gtk::Align::Center)
        .build();

    let search_bar = gtk::SearchBar::builder()
        .valign(gtk::Align::Start)
        .key_capture_widget(&window)
        .build();

    let search_button = gtk::ToggleButton::new();
    search_button.set_icon_name("system-search-symbolic");
    search_button
        .bind_property("active", &search_bar, "search-mode-enabled")
        .sync_create()
        .bidirectional()
        .build();

    let confirm_search = gtk::Button::builder()
        .label("Generate")
        .build();

    let entry = gtk::SearchEntry::new();
    entry.set_hexpand(true);
    search_bar.set_child(Some(&entry));
    search_button.set_active(true);

    entry.connect_search_changed(clone!(@weak label => move |entry| {
        // if entry.text() != "" {
        //     label.set_text(&entry.text());
        // } else {
        //     label.set_text("Type to start search");
        // }
    }));

    //----------------------

    // … with a button in it …
    let button = gtk::Button::with_label("Hello World!");
    // … which closes the window when clicked
    button.connect_clicked(clone!(@weak window => move |_| print_test()));

    let drawing_area = gtk::DrawingArea::new();
    drawing_area.set_size_request(512, 768);
    drawing_area.set_visible(true);

    let image_surface = gdk::cairo::ImageSurface::create(cairo::Format::ARgb32, 512, 768).unwrap();
    let surface = &*image_surface;
    app_state.borrow_mut().surface = Some(surface.clone());

    drawing_area.set_draw_func(
        clone!(@strong app_state => move |drawing_area, cr, width, height| {
            // Use AppState's surface
            if let Some(surface) = &app_state.borrow().surface {
                cr.set_source_surface(surface, 0.0, 0.0).unwrap();
                cr.paint().unwrap();
            }
        }),
    );

    drawing_area.connect_realize(clone!(@weak drawing_area, @weak app_state => move |_| {
        println!("Drawing area connected shown");
    pressed(512 as f64,768 as f64,&drawing_area, &app_state);
    drawing_area.queue_draw();
    }));

    let drag = gtk::GestureDrag::new();
    drag.set_button(1);

    drag.connect_drag_begin(clone!(@weak drawing_area,@weak app_state => move |_, x,y| {
            drag_begin( x, y, &drawing_area, &app_state);
    }));
    drag.connect_drag_update(clone!(@weak drawing_area,@weak app_state => move |_, x,y| {
            drag_update( x, y, &drawing_area, &app_state);
    }));
    drag.connect_drag_end(clone!(@weak drawing_area,@weak app_state => move |_, x,y| {
            //draw_brush( &drawing_area, x, y,  &app_state);
    }));

    let press = gtk::GestureClick::new();
    press.set_button(2);

    press.connect_pressed(
        clone!(@weak drawing_area,@weak app_state => move |gesture, button, x, y| {
            pressed(x as f64,y as f64,&drawing_area, &app_state);
        }),
    );

    drawing_area.add_controller(press);
    drawing_area.add_controller(drag);

    let drawing_area_image = gtk::DrawingArea::new();
    drawing_area_image.set_size_request(512, 768);
    drawing_area_image.set_visible(true);

    let image_surface_image =
        gdk::cairo::ImageSurface::create(cairo::Format::ARgb32, 512, 768).unwrap();
    let surface_image = &*image_surface_image;
    //picture.add_controller(motion_controller);

    // let grid = gtk::Grid::builder()
    // .margin_start(6)
    // .margin_end(6)
    // .margin_top(6)
    // .margin_bottom(6)
    // .halign(gtk::Align::Center)
    // .valign(gtk::Align::Center)
    // .row_spacing(6)
    // .column_spacing(6)
    // .build();
    let file_button = gtk::MenuButton::new();
    file_button.set_label("File");

    let ai_button = gtk::MenuButton::new();
    ai_button.set_label("AI Models");

    header_bar.pack_end(&ai_button);
    header_bar.pack_end(&file_button);
    header_bar.pack_end(&search_button);
    //grid.attach(&container, 0, 0, 5, 5);
    //grid.attach(&button, 0, 1, 1, 1);
    //grid.attach(&drawing_area, 0, 2, 2, 2);
    // -------------------------------------------
    let container_info = gtk::Box::new(gtk::Orientation::Vertical, 6);
    container_info.set_width_request(512);
    let info_label = gtk::Label::builder().halign(gtk::Align::Center).build();
    info_label.set_text("AI Image Generation Progress");

    let test_searchbar = gtk::ProgressBar::builder()
        .valign(gtk::Align::Center)
        .margin_end(12)
        .margin_start(12)
        .fraction(0.5)
        //.width_request(512)
        //.key_capture_widget(&window)
        .build();

    container_info.append(&info_label);
    container_info.append(&test_searchbar);

    //test_searchbar.set_child(Some(&text_entry));
    //    search_test.set_active(true);

    // -------------------------------------------
    // let progress_bar = gtk::ProgressBar::builder()
    //     .halign(gtk::Align::Fill)
    //     .hexpand(true)
    //     .visible(true)
    //     .can_target(true)
    //     .build();

    // progress_bar.set_text(Some("Generation progress"));
    // progress_bar.set_fraction(0.5);

    // --------------------------------------------

    let container = gtk::Box::new(gtk::Orientation::Horizontal, 6);

    let search_grid = gtk::Grid::builder()
        .halign(gtk::Align::Fill)
        .valign(gtk::Align::Baseline)
        .row_spacing(6)
        .build();

    search_grid.attach(&search_bar, 0, 0, 1, 1);
    search_grid.attach(&confirm_search, 1, 0, 1, 1);

    let full_grid = gtk::Grid::builder()
        .halign(gtk::Align::Fill)
        .valign(gtk::Align::Baseline)
        .row_spacing(6)
        .build();

    full_grid.attach(&search_grid, 0, 0, 1, 1);
    full_grid.attach(&drawing_area, 0, 1, 1, 1);

    let other_grid = gtk::Grid::builder()
        .halign(gtk::Align::Fill)
        .valign(gtk::Align::Baseline)
        .row_spacing(6)
        .build();

    other_grid.attach(&container_info, 0, 0, 1, 1);
    other_grid.attach(&drawing_area_image, 0, 1, 1, 1);
    // container.append(&label);
    // container.append(&grid);
    // //container.append(&button);
    // container.append(&drawing_area);
    container.append(&full_grid);
    container.append(&other_grid);
    // let win_box = gtk::Box::new(gtk::Orientation::Vertical, 12);
    // win_box.append(&button);
    // win_box.append(&drawing_area);

    // frame.set_child(Some(&win_box));
    //frame.set_child(Some(&grid));

    frame.set_child(Some(&container));
    window.set_child(Some(&frame));
    window.present();
}

fn print_test() {
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
