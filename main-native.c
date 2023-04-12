#include "capturer.h"
#include "main.h"
#include "renderer.h"

#include <bits/pthreadtypes.h>
#include <bits/types/struct_sched_param.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>

// get face data to use given current time (specified in pair->ms)
extern void face_get(time_face_t *pair);

long get_current_ms() {
    struct timespec spec;
    clock_gettime(CLOCK_MONOTONIC, &spec);
    return spec.tv_sec * 1e3 + spec.tv_nsec / 1e6;
}

void sleep_ms(int ms) {
    struct timespec req, remain;
    req.tv_sec = 0;
    req.tv_nsec = ms * 1e6;
    nanosleep(&req, &remain);
}

// does not terminate
static void *capturer_server(void *arg_p) {
    (void)arg_p;
    capturer_serve();
    return NULL;
}

// does not terminate
static void *capturer_getter(void *arg_p) {
    (void)arg_p;
    capturer_get();
    return NULL;
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        printf("Usage: ./373-native <camera> <focus> <puppet file>\n");
        return -1;
    }

    capturer_init(argv[1], atoi(argv[2]));
    renderer_init(argv[3]);

    pthread_t capturer_server_thread;
    pthread_create(&capturer_server_thread, NULL, capturer_server, NULL);

    pthread_attr_t attr;
    struct sched_param param;
    pthread_attr_init(&attr);
    pthread_attr_getschedparam(&attr, &param);
    // the neural network is the slowest and should be started ASAP
    // increasing priority to help capturer_get() get past block_until_face_needed()
    param.sched_priority = sched_get_priority_max(SCHED_FIFO);
    pthread_t capturer_getter_thread;
    pthread_create(&capturer_getter_thread, &attr, capturer_getter, NULL);

    sleep_ms(TIME_BUFFER);

    // also elevate the rendering thread (self)
    // FIFO is safe as we have 2 cores on target (and capturer_getter() is guaranteed to block)
    pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);

    while (1) {
        time_face_t pair;
        pair.ms = get_current_ms();
        face_get(&pair);
        renderer_render(&pair.face);
    }

    return 0;
}