#pragma once

#include "main.h"

// output(t) = realtime(t - TIMEBUFFER)
#define TIME_BUFFER 100

// use a moving average of 5 latest faces
// interpolates between [0] and [1] to get the output
// when [1] is passed, pops [0] and pushes new (to [0])
#define FACE_BUFFER 10

void capturer_init(char *cam_dev_name, int focus);

void capturer_serve();

void capturer_get();