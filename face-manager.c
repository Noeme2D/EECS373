#include "capturer.h"
#include "main.h"

#include <bits/pthreadtypes.h>
#include <math.h>
#include <pthread.h>
#include <time.h>

static const face_t DEFAULT_FACE = {0,      {0, 0, 0}, {0, 0, 0}, {1, 1}, {0, 0}, {0, 0},
                                    {0, 0}, {0, 0},    {0, 0},    {1, 1}, 0,      1};

typedef struct {
    int head; // points to the oldest face
    time_face_t raw_pairs[FACE_BUFFER];
    face_t moving_avg;
    face_t avg_faces[FACE_BUFFER];
} FaceManager;

static FaceManager fm;
static pthread_mutex_t fm_lock = PTHREAD_MUTEX_INITIALIZER;
static int need_new_face;
static pthread_cond_t need_new_face_cond = PTHREAD_COND_INITIALIZER;
static pthread_cond_t new_face_present_cond = PTHREAD_COND_INITIALIZER;

void face_init() {
    long current_ms = get_current_ms();
    for (int i = 0; i < FACE_BUFFER; i++) {
        fm.raw_pairs[i].ms = current_ms;
        fm.raw_pairs[i].face = DEFAULT_FACE;
        fm.avg_faces[i] = DEFAULT_FACE;
        current_ms += TIME_BUFFER;
    }
    fm.head = 0;
    fm.moving_avg = DEFAULT_FACE;
    need_new_face = 0;
}

void block_until_face_needed() {
    pthread_mutex_lock(&fm_lock);
    while (!need_new_face) {
        pthread_cond_wait(&need_new_face_cond, &fm_lock);
    }
    pthread_mutex_unlock(&fm_lock);
}

void face_update(time_face_t *pair) {
    pthread_mutex_lock(&fm_lock);

    if (pair->face.track_fail) {
        // if failure, copy the latest face
        int latest_head = (fm.head + FACE_BUFFER - 1) % FACE_BUFFER;
        pair->face = fm.avg_faces[latest_head];
    }

    fm.raw_pairs[fm.head].ms = pair->ms;
    for (int i = 0; i < 22; i++) {
        // starting from (float *)translation there are 22 floats
        fm.moving_avg.translation[i] +=
            (pair->face.translation[i] - fm.raw_pairs[fm.head].face.translation[i]) / FACE_BUFFER;
        fm.raw_pairs[fm.head].face.translation[i] = pair->face.translation[i];
        fm.avg_faces[fm.head].translation[i] = fm.moving_avg.translation[i];
    }

    fm.head = (fm.head + 1) % FACE_BUFFER;

    need_new_face = 0;
    pthread_cond_signal(&new_face_present_cond);

    pthread_mutex_unlock(&fm_lock);
}

void face_get(time_face_t *pair) {
    long ms = pair->ms - TIME_BUFFER;

    pthread_mutex_lock(&fm_lock);

    int left = fm.head;
    int right = (fm.head + 1) % FACE_BUFFER;

    // if required time has gone past the oldest interval
    if (ms > fm.raw_pairs[right].ms) {
        left = (left + 1) % FACE_BUFFER;
        right = (right + 1) % FACE_BUFFER;
        need_new_face = 1;
        pthread_cond_signal(&need_new_face_cond);
    }

    while (ms > fm.raw_pairs[right].ms) {
        left = (left + 1) % FACE_BUFFER;
        right = (right + 1) % FACE_BUFFER;

        if (right == fm.head) {
            // rendering has gone too far, must block until a new face
            while (need_new_face) {
                pthread_cond_wait(&new_face_present_cond, &fm_lock);
            }
            break;
        }
    }

    // linear interpolation
    float left_factor = (float)(fm.raw_pairs[right].ms - ms) / (fm.raw_pairs[right].ms - fm.raw_pairs[left].ms);
    float right_factor = (float)(ms - fm.raw_pairs[left].ms) / (fm.raw_pairs[right].ms - fm.raw_pairs[left].ms);
    for (int i = 0; i < 22; i++) {
        pair->face.translation[i] =
            fm.avg_faces[left].translation[i] * left_factor + fm.avg_faces[right].translation[i] * right_factor;
    }

    pthread_mutex_unlock(&fm_lock);
}