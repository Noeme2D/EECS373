#include "main.h"

#include <fcntl.h>
#include <linux/v4l2-controls.h>
#include <linux/videodev2.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/poll.h>
#include <unistd.h>

#define WIDTH 640
#define HEIGHT 480
#define BYTES_PER_PIXEL 2

int fd;

char *camera_init(char *device_name, int focus) {
    fd = open(device_name, O_RDWR);
    if (fd == -1) {
        perror("Opening video device");
        return NULL;
    }

    struct v4l2_format fmt = {0};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = WIDTH;
    fmt.fmt.pix.height = HEIGHT;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;
    if (ioctl(fd, VIDIOC_S_FMT, &fmt)) {
        perror("Setting Pixel Format");
        return NULL;
    }

    struct v4l2_ext_controls ctrls = {0};
    struct v4l2_ext_control ctrl = {0};

    ctrls.which = V4L2_CTRL_WHICH_CUR_VAL;
    ctrls.count = 1;
    ctrls.controls = &ctrl;
    ctrl.id = V4L2_CID_FOCUS_AUTO;
    ctrl.value = 0;
    if (ioctl(fd, VIDIOC_S_EXT_CTRLS, &ctrls)) {
        perror("Shutting Down Auto-Focus");
        return NULL;
    }

    ctrls.which = V4L2_CTRL_WHICH_CUR_VAL;
    ctrls.count = 1;
    ctrls.controls = &ctrl;
    ctrl.id = V4L2_CID_FOCUS_ABSOLUTE;
    ctrl.value = focus;
    if (ioctl(fd, VIDIOC_S_EXT_CTRLS, &ctrls)) {
        perror("Setting Focus");
        return NULL;
    }

    struct v4l2_requestbuffers req = {0};
    req.count = 1;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (ioctl(fd, VIDIOC_REQBUFS, &req)) {
        perror("Requesting Buffer");
        return NULL;
    }

    struct v4l2_buffer buf = {0};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = 0;
    if (ioctl(fd, VIDIOC_QUERYBUF, &buf)) {
        perror("Querying Buffer");
        return NULL;
    }

    return mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);
}

int camera_capture() {
    struct v4l2_buffer buf = {0};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = 0;
    if (ioctl(fd, VIDIOC_QBUF, &buf)) {
        perror("Query Buffer");
        return -1;
    }

    if (ioctl(fd, VIDIOC_STREAMON, &buf.type)) {
        perror("Start Capture");
        return -1;
    }

    struct pollfd pfd;
    pfd.fd = fd;
    pfd.events = POLLIN;
    if (poll(&pfd, 1, 50) == -1) {
        perror("Waiting for Frame");
        return -1;
    }

    if (ioctl(fd, VIDIOC_DQBUF, &buf)) {
        perror("Retrieving Frame");
        return -1;
    }

    return 0;
}