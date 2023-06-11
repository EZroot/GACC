#include "drawing_area.h"

G_DEFINE_TYPE(MyDrawingArea, my_drawing_area, GTK_TYPE_WIDGET)

static void my_drawing_area_snapshot(GtkWidget *widget, GtkSnapshot *snapshot) {
    MyDrawingArea *self = (MyDrawingArea *)widget;

    for (guint i = 0; i < self->points->len; i++) {
        Point point = g_array_index(self->points, Point, i);
        graphene_rect_t rect = GRAPHENE_RECT_INIT(point.x - 2, point.y - 2, 2, 2);
        GdkRGBA color = {0, 0, 0, 1};
        gtk_snapshot_append_color(snapshot, &color, &rect);
    }
}

static void my_drawing_area_on_event(GtkEventControllerMotion *controller, double x, double y, gpointer user_data) {
    MyDrawingArea *area = (MyDrawingArea *)user_data;

    // Only draw if button is pressed down
    if (area->button_down) {
        Point point = {x, y};
        g_array_append_val(area->points, point);
        gtk_widget_queue_draw(GTK_WIDGET(area));
    }
}

static void my_drawing_area_on_press(GtkGestureClick *gesture, int n_press, double x, double y, gpointer user_data) {
    MyDrawingArea *area = (MyDrawingArea *)user_data;
    area->button_down = TRUE;
}

static void my_drawing_area_on_release(GtkGestureClick *gesture, int n_press, double x, double y, gpointer user_data) {
    MyDrawingArea *area = (MyDrawingArea *)user_data;
    area->button_down = FALSE;
}

static void my_drawing_area_class_init(MyDrawingAreaClass *klass) {
    GtkWidgetClass *widget_class = GTK_WIDGET_CLASS(klass);
    widget_class->snapshot = my_drawing_area_snapshot;
}

static void my_drawing_area_init(MyDrawingArea *self) {
    self->points = g_array_new(FALSE, FALSE, sizeof(Point));
    self->button_down = FALSE; // Initial value

    // Add the motion event controller
    GtkEventController *motion_controller = gtk_event_controller_motion_new();
    g_signal_connect(motion_controller, "motion", G_CALLBACK(my_drawing_area_on_event), self);
    gtk_widget_add_controller(GTK_WIDGET(self), motion_controller);

    // Add the button gesture controller
    GtkGesture *button_gesture = gtk_gesture_click_new();
    g_signal_connect(button_gesture, "pressed", G_CALLBACK(my_drawing_area_on_press), self);
    g_signal_connect(button_gesture, "released", G_CALLBACK(my_drawing_area_on_release), self);
    gtk_widget_add_controller(GTK_WIDGET(self), GTK_EVENT_CONTROLLER(button_gesture));
}

GtkWidget *my_drawing_area_new(void) {
    return GTK_WIDGET(g_object_new(my_drawing_area_get_type(), NULL));
}
