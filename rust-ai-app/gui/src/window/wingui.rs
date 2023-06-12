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
use gtk::gdk_pixbuf::Pixbuf;
use gtk::graphene::ffi::{graphene_rect_init, graphene_rect_t};
use gtk::{gdk, glib, graphene, gsk, CssProvider, STYLE_PROVIDER_PRIORITY_APPLICATION};
use gtk::{prelude::*, ProgressBar};
use tokio::runtime::Builder;

use crate::window::appstates::AppState;

use super::components::airequestor::AIRequestor;
use super::components::gestures::Gestures;
use super::components::helper::Helper;

pub struct WindowsApp {
    title: String,
    size: (i32, i32),
    resizable: bool,
    force_dark_theme: bool,
}

impl WindowsApp {
    pub fn new(
        title: String,
        win_size: (i32, i32),
        resizable: bool,
        force_dark_theme: bool,
    ) -> Self {
        Self {
            title: title,
            size: win_size,
            resizable: resizable,
            force_dark_theme: force_dark_theme,
        }
    }

    // When the application is launched…
    pub async fn on_activate(&self, application: &gtk::Application) {
        // … create a new window …
        let window = self.create_new_window(application);

        let app_state = Rc::new(RefCell::new(AppState::new()));

        let frame = gtk::Frame::new(None);

        let prompt_bar = self.create_prompt_bar(&window, app_state.clone());

        let generate_image_button = self.create_button_generate_image(app_state.clone());

        let drawing_area = self.create_drawing_area(app_state.clone());
        self.initialize_drawing_gestures(&drawing_area, app_state.clone());

        let container_info = self.create_progress_bar_container();

        let drawing_area_loaded_image = self.create_drawing_area(app_state.clone());
        let image_filepath = "C:/Repos/ultimate-ai-assistant/python-ai-backend/gen_pics/2ebcff9bd6ddc2176342fcbe4dca0d1ce369bdba7c2f50ed8ab40ea21231ea0c.png".to_string();
        self.load_image_to_drawing_area(
            image_filepath,
            &drawing_area_loaded_image,
            app_state.clone(),
        );

        let container = self.create_container_append_widgets(container_info, prompt_bar, generate_image_button, drawing_area, drawing_area_loaded_image);
        // --------------------------------------------

        frame.set_child(Some(&container));
        window.set_child(Some(&frame));
        window.present();
    }

    fn create_new_window(&self, application: &gtk::Application) -> gtk::ApplicationWindow {
        let window = gtk::ApplicationWindow::new(application);
        window.set_title(Some(&self.title));
        window
            .settings()
            .set_gtk_application_prefer_dark_theme(self.force_dark_theme);
        window.set_default_size(self.size.0, self.size.1);
        window.set_resizable(self.resizable);
        window
    }

    fn create_prompt_bar(
        &self,
        window: &gtk::ApplicationWindow,
        app_state: Rc<RefCell<AppState>>,
    ) -> gtk::SearchBar {
        let header_bar = gtk::HeaderBar::new();
        window.set_titlebar(Some(&header_bar));

        let prompt_bar = gtk::SearchBar::builder()
            .valign(gtk::Align::Start)
            .key_capture_widget(window)
            .build();
        let show_prompt_bar = gtk::ToggleButton::new();
        show_prompt_bar.set_icon_name("system-search-symbolic");
        show_prompt_bar
            .bind_property("active", &prompt_bar, "search-mode-enabled")
            .sync_create()
            .bidirectional()
            .build();

        let entry = gtk::SearchEntry::new();
        entry.set_hexpand(true);
        prompt_bar.set_child(Some(&entry));
        show_prompt_bar.set_active(true);

        entry.connect_search_changed(clone!(@weak app_state => move |entry| {
            if entry.text() != "" {
                app_state.borrow_mut().prompt = entry.text().to_string();
            }
        }));

        let file_button = gtk::MenuButton::new();
        file_button.set_label("File");

        let ai_button = gtk::MenuButton::new();
        ai_button.set_label("AI Models");

        header_bar.pack_end(&ai_button);
        header_bar.pack_end(&file_button);
        header_bar.pack_end(&show_prompt_bar);

        prompt_bar
    }

    fn create_button_generate_image(&self, app_state: Rc<RefCell<AppState>>) -> gtk::Button {
        let generate_image_button = gtk::Button::builder().label("Generate").build();
        generate_image_button.connect_clicked(clone!(@weak app_state => move |x| {
            let rt = Builder::new_current_thread().enable_all().build().unwrap();
            let prompt = &app_state.borrow_mut().prompt;
            println!("Generating image... {}", &prompt);
            rt.block_on(async move {
                match AIRequestor::send_ai_prompt_request(&prompt, 1,512,768).await {
                    Ok(response) => {
                        println!("Response: {}", response);
                    }
                    Err(e) => {
                        println!("Request failed: {}", e);
                    }
                }
            });
        }));
        generate_image_button
    }

    fn create_drawing_area(&self, app_state: Rc<RefCell<AppState>>) -> gtk::DrawingArea {
        let drawing_area = gtk::DrawingArea::new();
        drawing_area.set_size_request(512, 768);
        drawing_area.set_visible(true);

        let image_surface =
            gdk::cairo::ImageSurface::create(cairo::Format::ARgb32, 512, 768).unwrap();
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
        Gestures::pressed(512 as f64,768 as f64,&drawing_area, &app_state);
        drawing_area.queue_draw();
        }));

        drawing_area
    }

    fn initialize_drawing_gestures(
        &self,
        drawing_area: &gtk::DrawingArea,
        app_state: Rc<RefCell<AppState>>,
    ) {
        let drag = gtk::GestureDrag::new();
        drag.set_button(1);

        drag.connect_drag_begin(clone!(@weak drawing_area,@weak app_state => move |_, x,y| {
            Gestures::drag_begin( x, y, &drawing_area, &app_state);
        }));
        drag.connect_drag_update(clone!(@weak drawing_area,@weak app_state => move |_, x,y| {
            Gestures::drag_update( x, y, &drawing_area, &app_state);
        }));
        drag.connect_drag_end(clone!(@weak drawing_area,@weak app_state => move |_, x,y| {
                //draw_brush( &drawing_area, x, y,  &app_state);
        }));

        let press = gtk::GestureClick::new();
        press.set_button(2);

        press.connect_pressed(
            clone!(@weak drawing_area,@weak app_state => move |gesture, button, x, y| {
                Gestures::pressed(x as f64,y as f64,&drawing_area, &app_state);
            }),
        );

        drawing_area.add_controller(press);
        drawing_area.add_controller(drag);
    }

    fn create_progress_bar_container(&self) -> gtk::Box {
        let container_info = gtk::Box::new(gtk::Orientation::Vertical, 6);
        let info_label = gtk::Label::builder().halign(gtk::Align::Center).build();
        info_label.set_text("AI Image Generation Progress");

        let progress_bar = gtk::ProgressBar::builder()
            .valign(gtk::Align::Center)
            .margin_end(12)
            .margin_start(12)
            .fraction(0.5)
            .build();

        container_info.append(&info_label);
        container_info.append(&progress_bar);

        let tick = move || {
            let time = Helper::current_time();
            info_label.set_text(&time);
            glib::Continue(true)
        };

        glib::timeout_add_seconds_local(1, tick);
        container_info
    }

    fn load_image_to_drawing_area(
        &self,
        filepath: String,
        drawing_area_loaded_image: &gtk::DrawingArea,
        app_state: Rc<RefCell<AppState>>,
    ) {
        let mut pixbuf_loaded = false;
        let original_pixbuf = gdk::gdk_pixbuf::Pixbuf::from_file(filepath);
        let resized_pixbuf = match original_pixbuf {
            Ok(pixbuf) => {
                pixbuf_loaded = true;
                println!("Image loaded successfully");
                pixbuf.scale_simple(768, 768, gdk::gdk_pixbuf::InterpType::Bilinear)
            }
            Err(_) => {
                // Handle the case when the Pixbuf fails to load
                pixbuf_loaded = false;
                println!("Image failed to load");
                // Return a dummy Pixbuf or any other action you want to take
                gdk::gdk_pixbuf::Pixbuf::new(gdk::gdk_pixbuf::Colorspace::Rgb, true, 8, 1, 1)
            }
        }
        .expect("Failed to resize image");

        drawing_area_loaded_image.set_draw_func(
            clone!(@strong app_state => move |drawing_area_image, cr, width, height| {
                // Use AppState's surface
                if let Some(surface) = &app_state.borrow().surface {
                    println!("Set draw function to pixbuf");
                cr.set_source_rgb(1.0, 1.0, 1.0); // Set background color
                cr.paint().unwrap();

                if pixbuf_loaded {
                    cr.set_source_pixbuf(&resized_pixbuf, 0.0, 0.0);
                    cr.paint().unwrap();
                } else {
                    // Draw a placeholder or any other action you want to take
                    cr.set_source_rgb(0.5, 0.5, 0.5);
                    cr.rectangle(0.0, 0.0, 512.0, 768.0);
                    cr.fill().unwrap();
                }
                }
            }),
        );

        drawing_area_loaded_image.connect_realize(
            clone!(@weak drawing_area_loaded_image, @weak app_state => move |_| {
                println!("Drawing area connected shown");
                Gestures::pressed(512 as f64,768 as f64,&drawing_area_loaded_image, &app_state);
                drawing_area_loaded_image.queue_draw();
            }),
        );
    }

    fn create_container_append_widgets(
        &self,
        container_info : gtk::Box,
        prompt_bar: gtk::SearchBar,
        generate_image_button: gtk::Button,
        drawing_area: gtk::DrawingArea,
        drawing_area_loaded_image: gtk::DrawingArea,
    ) -> gtk::Box {
        let container = gtk::Box::new(gtk::Orientation::Horizontal, 6);

        let search_grid = gtk::Grid::builder()
            .halign(gtk::Align::Fill)
            .valign(gtk::Align::Baseline)
            .row_spacing(6)
            .build();

        search_grid.attach(&prompt_bar, 0, 0, 1, 1);
        search_grid.attach(&generate_image_button, 1, 0, 1, 1);

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
        other_grid.attach(&drawing_area_loaded_image, 0, 1, 1, 1);

        container.append(&full_grid);
        container.append(&other_grid);
        container
    }

}
