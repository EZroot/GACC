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
use crate::window::appstates::AppState;
use gtk::gdk::{Display, Paintable};
use gtk::gdk_pixbuf::{self, Pixbuf};
use gtk::graphene::ffi::{graphene_rect_init, graphene_rect_t};
use gtk::{
    gdk, glib, graphene, gsk, CssProvider, FileChooser, Inhibit, ResponseType, StateFlags,
    STYLE_PROVIDER_PRIORITY_APPLICATION,
};
use gtk::{prelude::*, ProgressBar};
use serde::{Deserialize, Serialize};
use tokio::runtime::Builder;

use super::components::airequestor::AIRequestor;
use super::components::gestures::Gestures;
use super::components::helper::Helper;

#[derive(Debug, Deserialize, Serialize)]
struct Response {
    image_path: String,
}

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

        let drawing_size = app_state.borrow_mut().image_size;
        let drawing_area = self.create_drawing_area(app_state.clone(), drawing_size);
        self.initialize_drawing_gestures(&drawing_area, app_state.clone());

        let save_button = gtk::Button::builder().label("Save").build();
        save_button.connect_clicked(clone!(@weak app_state, @weak drawing_area => move |_| { WindowsApp::save_drawing_area_as_image(app_state, drawing_area); }));

        let drawing_area_loaded_image = self.create_drawing_area(app_state.clone(), drawing_size);
        let image_filepath = "C:/Repos/ultimate-ai-assistant/python-ai-backend/gen_pics/2ebcff9bd6ddc2176342fcbe4dca0d1ce369bdba7c2f50ed8ab40ea21231ea0c.png".to_string();
        WindowsApp::load_image_to_drawing_area(
            image_filepath,
            &drawing_area_loaded_image,
            app_state.clone(),
            false,
        );

        let window_ref = window.clone();
        let load_button = gtk::Button::builder().label("Load Drawing").build();
        load_button.connect_clicked(clone!(@weak app_state, @weak drawing_area, @weak drawing_area_loaded_image => move |_| { 
             if(app_state.borrow().load_image_as_mask){
                let app_state_clone = app_state.clone();
                 WindowsApp::select_image_to_load_in_drawing_area(&drawing_area_loaded_image, app_state, &window_ref); 
                 //Gestures::clear_surface(&app_state_clone.borrow(), (0.0,0.0,0.0));
                 //drawing_area.queue_draw();
            }
            else{
            WindowsApp::select_image_to_load_in_drawing_area(&drawing_area, app_state, &window_ref); 
            }
        }));

        let prompt_bar = self.create_prompt_bar(&window, app_state.clone());
        let generate_image_button =
            self.create_button_generate_image(app_state.clone(), drawing_area_loaded_image.clone());

        let container_info = self.create_progress_bar_container();
        let container = self.create_container_append_widgets(
            container_info,
            prompt_bar,
            generate_image_button,
            drawing_area,
            drawing_area_loaded_image,
        );
        container.append(&save_button);
        container.append(&load_button);

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

    pub fn save_drawing_area_as_image(
        app_state: Rc<RefCell<AppState>>,
        drawing_area: gtk::DrawingArea,
    ) {
        let question_dialog = gtk::Dialog::builder().modal(true).build();
        //question_dialog.add_buttons(["Cancel", "Ok"]);
        question_dialog.set_title(Some("Save file"));
        //let answer = question_dialog.choose_future(Some(&*window)).await;

        question_dialog.show();

        let app_state_clone = app_state.clone();

        let img_size = app_state_clone.borrow().image_size;

        if let Some(surface) = &app_state.borrow_mut().surface {
            let mut image_surface =
                cairo::ImageSurface::create(cairo::Format::ARgb32, img_size.0, img_size.1).unwrap();
            {
                let cr = cairo::Context::new(&image_surface).unwrap();
                cr.set_source_surface(surface, 0.0, 0.0).unwrap();
                cr.paint().unwrap();
                cr.save().unwrap();
            }
            let pixel_data: Vec<u8> = image_surface.data().unwrap().to_vec(); //cairo::ImageSurface::data(&mut image_surface).unwrap().to_vec();

// Create a new Vec<u8> to store the RGB pixel data
let mut rgb_pixel_data = Vec::with_capacity(pixel_data.len());

// Iterate over each pixel (3 channels per pixel)
for i in (0..pixel_data.len()).step_by(4) {
    // Read the BGR values
    let blue = pixel_data[i];
    let green = pixel_data[i + 1];
    let red = pixel_data[i + 2];
    let alpha = pixel_data[i + 3];

    // Convert to RGB by swapping the blue and red values
    rgb_pixel_data.push(red);
    rgb_pixel_data.push(green);
    rgb_pixel_data.push(blue);
    rgb_pixel_data.push(alpha);

}
let bytes = glib::Bytes::from(&rgb_pixel_data);

            let width = img_size.0;
            let channels = 3; // 3 for RGB, 4 for RGBA
            let bytes_per_channel = 1; // 1 byte per channel for 8 bits per channel
            let bytes_per_pixel = channels * bytes_per_channel;
            let rowstride = bytes_per_pixel * width;
            let pixbuf = gdk_pixbuf::Pixbuf::from_bytes(
                &bytes,
                gdk_pixbuf::Colorspace::Rgb,
                true,
                8,
                width,
                img_size.1,
                width*4,
            );

            pixbuf.savev("testpic.png", "png", &[]).unwrap();
        }
    }

    fn create_prompt_bar(
        &self,
        window: &gtk::ApplicationWindow,
        app_state: Rc<RefCell<AppState>>,
    ) -> gtk::SearchBar {
        let header_bar = gtk::HeaderBar::new();
        window.set_titlebar(Some(&header_bar));
        let mode_switch = gtk::Switch::new();
        let switch_label = gtk::Label::new(Some("512x768 / 768x768"));
        let app_state_img_mod = app_state.clone();
        mode_switch.connect_state_set(move |_, state| {
            if state == true {
                println!("Req: Image Size set - 768,768");
                app_state_img_mod.borrow_mut().image_size = (768, 768);
            } else {
                println!("Req: Image Size set - 512,768");
                app_state_img_mod.borrow_mut().image_size = (512, 768);
            }
            Inhibit::default() // Return `Inhibit` to allow event propagation
        });

        let image_counter = gtk::Entry::builder()
            .placeholder_text("1")
            .tooltip_text("How many images to generate")
            .build();

        let app_state_img_counter = app_state.clone();
        image_counter.connect_changed(move |entry| {
            let my_integer = entry.text().parse::<i32>().unwrap_or(1);
            println!("Img counter changed! {}", my_integer);
            app_state_img_counter.borrow_mut().image_count = my_integer;
        });
        let image_counter_label = gtk::Label::builder().label("Image Count = ").build();

        header_bar.pack_start(&mode_switch);
        header_bar.pack_start(&switch_label);
        header_bar.pack_start(&image_counter_label);
        header_bar.pack_start(&image_counter);

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

    fn create_button_generate_image(
        &self,
        app_state: Rc<RefCell<AppState>>,
        drawing_area_loaded_image: gtk::DrawingArea,
    ) -> gtk::Button {
        let generate_image_button = gtk::Button::builder().label("Generate").build();
        generate_image_button.connect_clicked(clone!(@weak app_state, @weak drawing_area_loaded_image => move |_| {
            let rt = Builder::new_current_thread().enable_all().build().unwrap();
            let app_state_clone = Rc::clone(&app_state);
            let prompt = app_state_clone.borrow().prompt.clone();
            let img_count = app_state_clone.borrow().image_count.clone();
            let img_size = app_state_clone.borrow().image_size.clone();
            println!("Generating image... {}", &prompt);
            let app_state_clone2 = Rc::clone(&app_state_clone);
            rt.block_on(async move {
                match AIRequestor::send_ai_prompt_request(&prompt, img_count, img_size.0, img_size.1, "/mnt/c/Repos/ultimate-ai-assistant/rust-ai-app/gui/testpic.png").await {
                    Ok(response) => {
                        println!("Response: {}", response);
                        let res: Response = serde_json::from_str(&response).unwrap();
                        let win_path = res.image_path.replace("/mnt/c/", "C:/");
                        println!("Windows Path: {:?}", win_path);
                        WindowsApp::load_image_to_drawing_area(win_path, &drawing_area_loaded_image, app_state_clone2, false);
                    }
                    Err(e) => {
                        println!("Request failed: {}", e);
                    }
                }
            });
        }));
        generate_image_button
    }

    fn create_drawing_area(
        &self,
        app_state: Rc<RefCell<AppState>>,
        size: (i32, i32),
    ) -> gtk::DrawingArea {
        let drawing_area = gtk::DrawingArea::new();
        drawing_area.set_size_request(size.0, size.1);
        drawing_area.set_visible(true);

        let image_surface =
            gdk::cairo::ImageSurface::create(cairo::Format::ARgb32, size.0, size.1).unwrap();
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
        Gestures::pressed(size.0 as f64,size.1 as f64,&drawing_area, &app_state, (1.0,1.0,1.0));
        drawing_area.queue_draw();
        }));

        drawing_area
    }

    fn initialize_drawing_gestures(
        &self,
        drawing_area: &gtk::DrawingArea,
        app_state: Rc<RefCell<AppState>>,
    ) {
        let drag_left = gtk::GestureDrag::new();
        drag_left.set_button(1);

        drag_left.connect_drag_begin(clone!(@weak drawing_area,@weak app_state => move |_, x,y| {
            Gestures::drag_begin( x, y, &drawing_area, &app_state);
        }));
        drag_left.connect_drag_update(clone!(@weak drawing_area,@weak app_state => move |_, x,y| {
            Gestures::drag_update( x, y, &drawing_area, &app_state, (0.0,0.0,0.0));
        }));
        drag_left.connect_drag_end(clone!(@weak drawing_area,@weak app_state => move |_, x,y| {
                //draw_brush( &drawing_area, x, y,  &app_state);
        }));

        let drag_right = gtk::GestureDrag::new();
        drag_right.set_button(3);

        drag_right.connect_drag_begin(clone!(@weak drawing_area,@weak app_state => move |_, x,y| {
            Gestures::drag_begin( x, y, &drawing_area, &app_state);
        }));
        drag_right.connect_drag_update(clone!(@weak drawing_area,@weak app_state => move |_, x,y| {
            Gestures::drag_update( x, y, &drawing_area, &app_state, (1.0,1.0,1.0));
        }));
        drag_right.connect_drag_end(clone!(@weak drawing_area,@weak app_state => move |_, x,y| {
                //draw_brush( &drawing_area, x, y,  &app_state);
        }));

        let press = gtk::GestureClick::new();
        press.set_button(2);

        press.connect_pressed(
            clone!(@weak drawing_area,@weak app_state => move |gesture, button, x, y| {
                Gestures::pressed(x as f64,y as f64,&drawing_area, &app_state, (1.0,1.0,1.0));
            }),
        );

        drawing_area.add_controller(press);
        drawing_area.add_controller(drag_left);
        drawing_area.add_controller(drag_right);
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

    pub fn select_image_to_load_in_drawing_area(
        drawing_area_loaded_image: &gtk::DrawingArea,
        app_state: Rc<RefCell<AppState>>,
        window: &gtk::ApplicationWindow,
    ) {
        let file_chooser = gtk::FileChooserDialog::new(
            Some("Choose Image"),
            Some(window),
            gtk::FileChooserAction::Open,
            &[
                ("OK", ResponseType::Accept),
                ("Cancel", ResponseType::Cancel),
            ],
        );
        file_chooser.set_modal(true);

        file_chooser.run_async(clone!(@weak file_chooser,@weak drawing_area_loaded_image => move |x,y| {
                if y == ResponseType::Accept{
                    let file_path = &file_chooser.file().unwrap().path().unwrap();
                    let file_path_os = file_path.as_os_str().to_string_lossy();
                    let file_path_string = file_path_os.to_string();
                    println!("File accepted: {}",&file_path_os);
                    WindowsApp::load_image_to_drawing_area(file_path_string, &drawing_area_loaded_image, app_state, true);
                    file_chooser.close();
                }
                else if y == ResponseType::Cancel{
                    println!("Image load canceled");
                    file_chooser.close();
                }
            }));
    }

    pub fn load_image_to_drawing_area(
        filepath: String,
        drawing_area_loaded_image: &gtk::DrawingArea,
        app_state: Rc<RefCell<AppState>>,
        force_resize: bool,
    ) {
        let mut pixbuf_loaded = false;
        let original_pixbuf = gdk::gdk_pixbuf::Pixbuf::from_file(filepath);
        let resized_pixbuf = match original_pixbuf {
            Ok(pixbuf) => {
                pixbuf_loaded = true;
                println!("Image loaded successfully");
                if force_resize {
                    let ref_pixbuf = pixbuf
                        .scale_simple(512, 768, gdk::gdk_pixbuf::InterpType::Bilinear)
                        .unwrap();
                    Some(ref_pixbuf)
                } else {
                    Some(pixbuf)
                }
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

        let image_size = (resized_pixbuf.width(), resized_pixbuf.height()); //we dont want the app size here, we want to get the image size since thats what were loading...app_state.borrow_mut().image_size;

        drawing_area_loaded_image.set_size_request(image_size.0, image_size.1);

        //when we force resize, were assuming its going on the left drawing area
        //because why would we force a resize unless we want to fit it on our palleette
        //fix it later, just a note
        if force_resize==true {
            if let Some(surface) = &app_state.borrow_mut().surface {
                let crp = Context::new(surface).unwrap();
                println!("Set draw function to pixbuf");
                crp.set_source_rgb(1.0, 1.0, 1.0); // Set background color
                crp.paint().unwrap();
                if pixbuf_loaded {
                    crp.set_source_pixbuf(&resized_pixbuf, 0.0, 0.0);
                    crp.paint().unwrap();
                } else {
                    // Draw a placeholder or any other action you want to take
                    crp.set_source_rgb(0.5, 0.5, 0.5);
                    crp.rectangle(0.0, 0.0, 512.0, 768.0);
                    crp.fill().unwrap();
                }
                crp.save().unwrap();
            }
        }else{
        drawing_area_loaded_image.set_draw_func(
            clone!(@weak app_state => move |drawing_area_image, cr, width, height| {
                // Use AppState's surface
                println!("Set draw function to pixbuf");
                cr.set_source_rgb(1.0, 1.0, 1.0); // Set background color
                cr.paint().unwrap();

                if pixbuf_loaded {
                    cr.set_source_pixbuf(&resized_pixbuf, 0.0, 0.0);
                    cr.paint().unwrap();
                } else {
                    // Draw a placeholder or any other action you want to take
                    cr.set_source_rgb(0.5, 0.5, 0.5);
                    cr.rectangle(0.0, 0.0, 768.0, 768.0);
                    cr.fill().unwrap();

                }
                cr.save().unwrap();
            }),

        );
    }
        drawing_area_loaded_image.connect_realize(
            clone!(@weak drawing_area_loaded_image, @weak app_state => move |_| {
                println!("Drawing area connected shown");
                let image_size = app_state.borrow_mut().image_size;
                Gestures::pressed(image_size.0 as f64,image_size.1 as f64,&drawing_area_loaded_image, &app_state, (1.0,1.0,1.0));
                drawing_area_loaded_image.queue_draw();
            }),
        );

        drawing_area_loaded_image.queue_draw();
    }

    fn create_container_append_widgets(
        &self,
        container_info: gtk::Box,
        prompt_bar: gtk::SearchBar,
        generate_image_button: gtk::Button,
        drawing_area: gtk::DrawingArea,
        drawing_area_loaded_image: gtk::DrawingArea,
    ) -> gtk::Box {
        let container = gtk::Box::new(gtk::Orientation::Vertical, 6);

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
        full_grid.attach(&container_info, 1, 0, 1, 1);

        let other_grid = gtk::Grid::builder()
            .halign(gtk::Align::Fill)
            .valign(gtk::Align::Baseline)
            .row_spacing(6)
            .build();

        other_grid.attach(&drawing_area, 0, 0, 1, 1);
        other_grid.attach(&drawing_area_loaded_image, 1, 0, 1, 1);

        container.append(&full_grid);
        container.append(&other_grid);
        container
    }
}
