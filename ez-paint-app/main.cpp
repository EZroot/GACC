#include <gtk/gtk.h>
#include "components/drawing_area.h"

static void activate(GtkApplication* app, gpointer user_data);


int main(int argc, char** argv) {
    GtkApplication* app;
    int status;

    app = gtk_application_new("org.gtk.example", G_APPLICATION_DEFAULT_FLAGS);
    g_signal_connect(app, "activate", G_CALLBACK(activate), NULL);
    status = g_application_run(G_APPLICATION(app), argc, argv);
    g_object_unref(app);

    return status;
}

static void print_hello(GtkWidget* widget, gpointer data) {
    g_print("Hello World\n");
}

static gboolean on_draw(GtkWidget *widget, cairo_t *cr, gpointer data) {
    // Set the color and line width for the rectangle outline
    cairo_set_source_rgb(cr, 0, 0, 0);  // Black color
    cairo_set_line_width(cr, 2.0);     // Line width of 2 pixels

    // Get the size of the drawing area
    int width, height;
    gtk_widget_get_size_request(widget, &width, &height);

    // Draw the outline rectangle
    cairo_rectangle(cr, 0, 0, width, height);
    cairo_stroke(cr);

    return FALSE;
}

static void activate(GtkApplication* app, gpointer user_data) {
    GtkWidget* window;
    GtkWidget* drawing_area;
    GtkWidget* button;

    window = gtk_application_window_new(app);
    gtk_window_set_title(GTK_WINDOW(window), "Window");

    drawing_area = my_drawing_area_new();
    gtk_widget_set_size_request(drawing_area, 512, 768);

    button = gtk_button_new_with_label("Click Me");
    g_signal_connect(button, "clicked", G_CALLBACK(print_hello), NULL);
    // Create a vertical box and pack the button and drawing area into it
    GtkWidget* box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    gtk_box_append(GTK_BOX(box), button);
    gtk_box_append(GTK_BOX(box), drawing_area);
    gtk_window_set_child(GTK_WINDOW(window), box);

    //gtk_window_set_child(GTK_WINDOW(window), drawing_area);
    gtk_window_present(GTK_WINDOW(window));
}