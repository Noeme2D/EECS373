#pragma once

#include <bits/time.h>
#include <time.h>

typedef struct {
    int track_fail;
    float translation[3];
    float rotation[3];
    float eye[2];
    float eye_blink[2];
    float eyebrow_steepness[2];
    float eyebrow_quirk[2];
    float eyebrow_down[2];
    float mouth_corner_down[2];
    float mouth_corner_inout[2];
    float mouth_open;
    float mouth_wide;
} face_t;

typedef struct {
    long ms;
    face_t face;
} time_face_t;

long get_current_ms();

void sleep_ms(int);