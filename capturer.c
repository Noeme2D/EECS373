#include "capturer.h"
#include "main.h"

static const face_t DEFAULT_FACE = {0,      {0, 0, 0}, {0, 0, 0}, {1, 1}, {0, 0}, {0, 0},
                                    {0, 0}, {0, 0},    {0, 0},    {1, 1}, 0,      1};

// returns mmap region the camera will write into
extern char *camera_init(char *device_name, int focus);

// takes a picture (< 1ms)
extern int camera_capture();

static char *img_buffer;

// inits face_manager
extern void face_init();

// MUST start getting a new face and run face_update() soon after return
extern void block_until_face_needed();

// updates face_manager with latest data
extern void face_update(time_face_t *pair);

// loads python
static void osf_init() {
}

// runs NN inference once
static void osf_run(char *yuyv_img, face_t *face) {
    sleep_ms(50);

    face->track_fail = 0;
    face->eye[0] = get_current_ms();
    face->eye[1] = 5.0;
}

void capturer_init(char *cam_dev_name, int focus) {
    img_buffer = camera_init(cam_dev_name, focus);
    osf_init();
    face_init();
}

void capturer_serve() {
    while (1)
        ;
}

void capturer_get() {
    time_face_t pair;

    while (1) {
        block_until_face_needed();
        pair.ms = get_current_ms();

        if (camera_capture()) {
            // camera failed
            pair.face = DEFAULT_FACE;
            sleep_ms(50);
        } else {
            osf_run(img_buffer, &pair.face);
        }

        face_update(&pair);
    }
}