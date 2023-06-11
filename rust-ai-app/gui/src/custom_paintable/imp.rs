use gtk::gdk::ffi::GdkRGBA;
use gtk::prelude::*;
use gtk::subclass::prelude::*;
use gtk::{gdk, glib, graphene, gsk};

#[derive(Default)]
pub struct CustomPaintable {
    points: Vec<(f32, f32)>,
}

#[glib::object_subclass]
impl ObjectSubclass for CustomPaintable {
    const NAME: &'static str = "CustomPaintable";
    type Type = super::CustomPaintable;
    type Interfaces = (gdk::Paintable,);
}

impl ObjectImpl for CustomPaintable {}

impl PaintableImpl for CustomPaintable {
    fn flags(&self) -> gdk::PaintableFlags {
        // Fixed size
        gdk::PaintableFlags::SIZE
    }

    fn intrinsic_width(&self) -> i32 {
        512
    }

    fn intrinsic_height(&self) -> i32 {
        768
    }

    fn snapshot(&self, snapshot: &gdk::Snapshot, width: f64, height: f64) {
        for point in &self.points {
            let pixel = gtk::graphene::Rect::new(point.0 as f32, point.1 as f32, 8.0, 8.0);
            let color = gdk::RGBA::new(0.0, 0.0, 0.0, 1.0);
            snapshot.append_color(&color, &pixel);
        }
    }
}