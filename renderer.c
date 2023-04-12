#include "renderer.h"
#include "main.h"

#include <stdio.h>
#include <time.h>

void renderer_init(char *filename) {
    printf("Puppet file: %s loaded.\n", filename);
}

void renderer_render(face_t *face) {
    // simulates 60 fps
    sleep_ms(16);

    printf("%.2f, %.2f", face->eye[0], face->eye[1]);
    printf("\r");
    fflush(stdout);
}