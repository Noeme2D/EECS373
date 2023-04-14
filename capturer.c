#include "capturer.h"
#include "main.h"

#include "osflite.h"

#include <netinet/in.h>
#include <sys/socket.h>

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
    printf("Loading libpython.\n");
    int err = PyImport_AppendInittab("osflite", PyInit_osflite);
    if (err) {
        printf("PyImport failed.\n");
    }
    Py_Initialize();
    PyImport_ImportModule("osflite");

    // Release Python Global Interpreter Lock
    PyEval_SaveThread();
}

// runs NN inference once
static void osf_run(char *yuyv_img, face_t *face) {
    run_osf(yuyv_img, face);
}

void capturer_init(char *cam_dev_name, int focus) {
    img_buffer = camera_init(cam_dev_name, focus);
    osf_init();
    face_init();
}

#define PORT 39539

void capturer_serve() {
    struct sockaddr_in from;
    from.sin_family = AF_INET;
    // only listen to local for demo purposes
    from.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    from.sin_port = htons(PORT);

    int conn = socket(AF_INET, SOCK_STREAM, 0);
    const int enable = 1;
    if (setsockopt(conn, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int)) < 0) {
        printf("setsockopt() failed.\n");
        exit(-1);
    }

    if (bind(conn, (struct sockaddr *)&from, sizeof(from)) < 0) {
        printf("Port bind failed.\n");
        exit(-1);
    }

    listen(conn, 1);
    // only accept one connection for demo purposes
    int incoming = accept(conn, 0, 0);

    // Acquire Python Global Interpreter Lock
    PyGILState_STATE gstate = PyGILState_Ensure();

    while (1) {
        char byte;
        // act upon one byte for demo purposes
        ssize_t recvd = recv(incoming, &byte, 1, 0);
        if (recvd <= 0) {
            printf("recv() of server failed.\n");
            exit(-1);
        }

        time_face_t pair;
        pair.ms = get_current_ms();
        if (camera_capture()) {
            printf("Camera failed.\n");
            pair.face = DEFAULT_FACE;
            sleep_ms(50);
        } else {
            osf_run(img_buffer, &pair.face);
            if (pair.face.track_fail) {
                printf("Face tracking failed.\n");
            }
        }

        // send the new face over network
        size_t bytes_sent = 0;
        do {
            // cares nothing about endianess for demo purposes
            ssize_t new_bytes_sent = send(incoming, (char *)(&pair) + bytes_sent, sizeof(time_face_t) - bytes_sent, 0);
            if (new_bytes_sent <= 0) {
                printf("Server send() failed.\n");
                exit(-1);
            }

            bytes_sent += new_bytes_sent;
        } while (bytes_sent < sizeof(time_face_t));
    }

    PyGILState_Release(gstate);
}

void capturer_get() {
    struct sockaddr_in to;
    to.sin_family = AF_INET;
    to.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    to.sin_port = htons(PORT);

    int conn = socket(AF_INET, SOCK_STREAM, 0);
    while (connect(conn, (struct sockaddr *)&to, sizeof(struct sockaddr))) {
        sleep_ms(1);
    }

    while (1) {
        block_until_face_needed();

        time_face_t pair;
        char req = 'G';
        // sends a char to the server to get a face
        if (send(conn, &req, 1, 0) < 0) {
            printf("Client send() failed.\n");
            exit(-1);
        }

        size_t bytes_rcvd = 0;
        do {
            ssize_t new_bytes_rcvd = recv(conn, (char *)(&pair) + bytes_rcvd, sizeof(time_face_t) - bytes_rcvd, 0);
            if (new_bytes_rcvd <= 0) {
                printf("Client rcvd() failed.\n");
                exit(-1);
            }

            bytes_rcvd += new_bytes_rcvd;
        } while (bytes_rcvd < sizeof(time_face_t));

        face_update(&pair);
    }
}