#ifndef MY_DRAWING_AREA_H
#define MY_DRAWING_AREA_H

#include <gtk/gtk.h>

typedef struct _MyDrawingArea MyDrawingArea;
typedef struct _MyDrawingAreaClass MyDrawingAreaClass;

struct _MyDrawingArea {
    GtkWidget parent_instance;
    GArray *points;
    gboolean button_down;
};

struct _MyDrawingAreaClass {
    GtkWidgetClass parent_class;
};

typedef struct {
    double x;
    double y;
} Point;

GtkWidget* my_drawing_area_new(void);

#endif // MY_DRAWING_AREA_H
