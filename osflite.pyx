import numpy as np
print("Numpy loaded.")
cimport numpy as np
import cv2
print("OpenCV loaded.")
import onnxruntime
print("ONNX Runtime loaded.")

# Constants
DEF WIDTH = 640
DEF HEIGHT = 480
DEF MODEL_RES = 224

# Image normalization constants
mean = np.float32(np.array([0.485, 0.456, 0.406]))
std = np.float32(np.array([0.229, 0.224, 0.225]))
mean = mean / std
std = std * 255.0
mean = -mean
std = 1.0 / std
mean_224 = np.tile(mean, [224, 224, 1])
std_224 = np.tile(std, [224, 224, 1])

camera = np.array([[WIDTH, 0, WIDTH/2], [0, WIDTH, HEIGHT/2], [0, 0, 1]], np.float32)
inverse_camera = np.linalg.inv(camera)

dist_coeffs = np.zeros((4,1))

# PnP Solving
face_3d = np.array([
    [ 0.4551769692672  ,  0.300895790030204, -0.764429433974752],
    [ 0.448998827123556,  0.166995837790733, -0.765143004071253],
    [ 0.437431554952677,  0.022655479179981, -0.739267175112735],
    [ 0.415033422928434, -0.088941454648772, -0.747947437846473],
    [ 0.389123587370091, -0.232380029794684, -0.704788385327458],
    [ 0.334630113904382, -0.361265387599081, -0.615587579236862],
    [ 0.263725112132858, -0.460009725616771, -0.491479221041573],
    [ 0.16241621322721 , -0.558037146073869, -0.339445180872282],
    [ 0.               , -0.621079019321682, -0.287294770748887],
    [-0.16241621322721 , -0.558037146073869, -0.339445180872282],
    [-0.263725112132858, -0.460009725616771, -0.491479221041573],
    [-0.334630113904382, -0.361265387599081, -0.615587579236862],
    [-0.389123587370091, -0.232380029794684, -0.704788385327458],
    [-0.415033422928434, -0.088941454648772, -0.747947437846473],
    [-0.437431554952677,  0.022655479179981, -0.739267175112735],
    [-0.448998827123556,  0.166995837790733, -0.765143004071253],
    [-0.4551769692672  ,  0.300895790030204, -0.764429433974752],
    [ 0.385529968662985,  0.402800553948697, -0.310031082540741],
    [ 0.322196658344302,  0.464439136821772, -0.250558059367669],
    [ 0.25409760441282 ,  0.46420381416882 , -0.208177722146526],
    [ 0.186875436782135,  0.44706071961879 , -0.145299823706503],
    [ 0.120880983543622,  0.423566314072968, -0.110757158774771],
    [-0.120880983543622,  0.423566314072968, -0.110757158774771],
    [-0.186875436782135,  0.44706071961879 , -0.145299823706503],
    [-0.25409760441282 ,  0.46420381416882 , -0.208177722146526],
    [-0.322196658344302,  0.464439136821772, -0.250558059367669],
    [-0.385529968662985,  0.402800553948697, -0.310031082540741],
    [ 0.               ,  0.293332603215811, -0.137582088779393],
    [ 0.               ,  0.194828701837823, -0.069158109325951],
    [ 0.               ,  0.103844017393155, -0.009151819844964],
    [ 0.               ,  0.               ,  0.               ],
    [ 0.080626352317973, -0.041276068128093, -0.134161035564826],
    [ 0.046439347377934, -0.057675223874769, -0.102990627164664],
    [ 0.               , -0.068753126205604, -0.090545348482397],
    [-0.046439347377934, -0.057675223874769, -0.102990627164664],
    [-0.080626352317973, -0.041276068128093, -0.134161035564826],
    [ 0.315905195966084,  0.298337502555443, -0.285107407636464],
    [ 0.275252345439353,  0.312721904921771, -0.244558251170671],
    [ 0.176394511553111,  0.311907184376107, -0.219205360345231],
    [ 0.131229723798772,  0.284447361805627, -0.234239149487417],
    [ 0.184124948330084,  0.260179585304867, -0.226590776513707],
    [ 0.279433549294448,  0.267363071770222, -0.248441437111633],
    [-0.131229723798772,  0.284447361805627, -0.234239149487417],
    [-0.176394511553111,  0.311907184376107, -0.219205360345231],
    [-0.275252345439353,  0.312721904921771, -0.244558251170671],
    [-0.315905195966084,  0.298337502555443, -0.285107407636464],
    [-0.279433549294448,  0.267363071770222, -0.248441437111633],
    [-0.184124948330084,  0.260179585304867, -0.226590776513707],
    [ 0.121155252430729, -0.208988660580347, -0.160606287940521],
    [ 0.041356305910044, -0.194484199722098, -0.096159882202821],
    [ 0.               , -0.205180167345702, -0.083299217789729],
    [-0.041356305910044, -0.194484199722098, -0.096159882202821],
    [-0.121155252430729, -0.208988660580347, -0.160606287940521],
    [-0.132325402795928, -0.290857984604968, -0.187067868218105],
    [-0.064137791831655, -0.325377847425684, -0.158924039726607],
    [ 0.               , -0.343742581679188, -0.113925986025684],
    [ 0.064137791831655, -0.325377847425684, -0.158924039726607],
    [ 0.132325402795928, -0.290857984604968, -0.187067868218105],
    [ 0.181481567104525, -0.243239316141725, -0.231284988892766],
    [ 0.083999507750469, -0.239717753728704, -0.155256465640701],
    [ 0.               , -0.256058040176369, -0.0950619498899  ],
    [-0.083999507750469, -0.239717753728704, -0.155256465640701],
    [-0.181481567104525, -0.243239316141725, -0.231284988892766],
    [-0.074036069749345, -0.250689938345682, -0.177346470406188],
    [ 0.               , -0.264945854681568, -0.112349967428413],
    [ 0.074036069749345, -0.250689938345682, -0.177346470406188],
    # Pupils and eyeball centers
    [ 0.257990002632141,  0.276080012321472, -0.219998998939991],
    [-0.257990002632141,  0.276080012321472, -0.219998998939991],
    [ 0.257990002632141,  0.276080012321472, -0.324570998549461],
    [-0.257990002632141,  0.276080012321472, -0.324570998549461]
], np.float32)

base_scale_v = face_3d[27:30, 1] - face_3d[28:31, 1]
base_scale_h = np.abs(face_3d[[0, 36, 42], 0] - face_3d[[16, 39, 45], 0])

# Constants

# Global objects
class Face_Info():
    def __init__(self):
        self.face_3d = face_3d
        self.rotation = np.array([0, 0, 0], np.float32)
        self.translation =  np.array([0, 0, 0], np.float32)
        self.contour_pts = [0,1,8,15,16,27,28,29,30,31,32,33,34,35]

face_info = Face_Info()
# Global objects

# Settings

detection_threshold = 0.6
model_type = 2

res = MODEL_RES
mean_res = mean_224
std_res = std_224

out_res = 27.0
out_res_i = 28

logit_factor = 16.0

# Settings

# Models setup

options = onnxruntime.SessionOptions()
options.inter_op_num_threads = 1
options.intra_op_num_threads = 1
options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
options.log_severity_level = 3
providersList = onnxruntime.capi._pybind_state.get_available_providers()

detection_model = onnxruntime.InferenceSession("optimized_detection_model.onnx", sess_options=options, providers=providersList)
lm_model = onnxruntime.InferenceSession("optimized_lm_model2.onnx", sess_options=options, providers=providersList)
lm_input_name = lm_model.get_inputs()[0].name 

# Models setup

# Utilities

def clamp_to_im(pt, w, h):
    x = pt[0]
    y = pt[1]
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x >= w:
        x = w-1
    if y >= h:
        y = h-1
    return (int(x), int(y+1))

def logit_arr(p, factor=16.0):
    p = np.clip(p, 0.0000001, 0.9999999)
    return np.log(p / (1 - p)) / float(factor)

def matrix_to_quaternion(m):
    t = 0.0
    q = [0.0, 0.0, 0, 0.0]
    if m[2,2] < 0:
        if m[0,0] > m[1,1]:
            t = 1 + m[0,0] - m[1,1] - m[2,2]
            q = [t, m[0,1]+m[1,0], m[2,0]+m[0,2], m[1,2]-m[2,1]]
        else:
            t = 1 - m[0,0] + m[1,1] - m[2,2]
            q = [m[0,1]+m[1,0], t, m[1,2]+m[2,1], m[2,0]-m[0,2]]
    else:
        if m[0,0] < -m[1,1]:
            t = 1 - m[0,0] - m[1,1] + m[2,2]
            q = [m[2,0]+m[0,2], m[1,2]+m[2,1], t, m[0,1]-m[1,0]]
        else:
            t = 1 + m[0,0] + m[1,1] + m[2,2]
            q = [m[1,2]-m[2,1], m[2,0]-m[0,2], m[0,1]-m[1,0], t]
    q = np.array(q, np.float32) * 0.5 / np.sqrt(t)
    return q

def angle(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    a = np.arctan2(*(p2 - p1)[::-1])
    return (a % (2 * np.pi))

def normalize_pts3d(pts_3d):
    # Calculate angle using nose
    pts_3d[:, 0:2] -= pts_3d[30, 0:2]
    alpha = angle(pts_3d[30, 0:2], pts_3d[27, 0:2])
    alpha -= np.deg2rad(90)

    R = np.matrix([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
    pts_3d[:, 0:2] = (pts_3d - pts_3d[30])[:, 0:2].dot(R) + pts_3d[30, 0:2]

    # Vertical scale
    pts_3d[:, 1] /= np.mean((pts_3d[27:30, 1] - pts_3d[28:31, 1]) / base_scale_v)

    # Horizontal scale
    pts_3d[:, 0] /= np.mean(np.abs(pts_3d[[0, 36, 42], 0] - pts_3d[[16, 39, 45], 0]) / base_scale_h)

    return pts_3d

def rotate(origin, point, a):
    a = -a
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(a) * (px - ox) - np.sin(a) * (py - oy)
    qy = oy + np.sin(a) * (px - ox) + np.cos(a) * (py - oy)
    return qx, qy

def align_points(a, b, pts):
    a = tuple(a)
    b = tuple(b)
    alpha = angle(a, b)
    alpha = np.rad2deg(alpha)
    if alpha >= 90:
        alpha = - (alpha - 180)
    if alpha <= -90:
        alpha = - (alpha + 180)
    alpha = np.deg2rad(alpha)
    aligned_pts = []
    for pt in pts:
        aligned_pts.append(np.array(rotate(a, pt, alpha)))
    return alpha, np.array(aligned_pts)

# Core

cdef extern from "main.h":
    ctypedef struct face_t:
        int track_fail
        float translation[3]
        float rotation[3]
        float eye[2]
        float eye_blink[2]
        float eyebrow_steepness[2]
        float eyebrow_quirk[2]
        float eyebrow_down[2]
        float mouth_corner_down[2]
        float mouth_corner_inout[2]
        float mouth_open
        float mouth_wide

def solve_features(pts):
    features = np.empty(16)
    norm_distance_x = np.mean([pts[0, 0] - pts[16, 0], pts[1, 0] - pts[15, 0]])
    norm_distance_y = np.mean([pts[27, 1] - pts[28, 1], pts[28, 1] - pts[29, 1], pts[29, 1] - pts[30, 1]])

    # 00 eye_l
    a1, f_pts = align_points(pts[42], pts[45], pts[[43, 44, 47, 46]])
    features[0] = abs((np.mean([f_pts[0,1], f_pts[1,1]]) - np.mean([f_pts[2,1], f_pts[3,1]])) / norm_distance_y)
    # 01 eye_r
    a2, f_pts = align_points(pts[36], pts[39], pts[[37, 38, 41, 40]]) 
    features[1] = abs((np.mean([f_pts[0,1], f_pts[1,1]]) - np.mean([f_pts[2,1], f_pts[3,1]])) / norm_distance_y)
    
    a3, _ = align_points(pts[0], pts[16], [])
    a4, _ = align_points(pts[31], pts[35], [])
    norm_angle = np.rad2deg(np.mean([a1, a2, a3, a4]))

    # 02 eye_blink_l
    features[2] = 1 - min(max(0, -features[0]), 1)
    # 03 eye_blink_r
    features[3] = 1 - min(max(0, -features[1]), 1)

    a, f_pts = align_points(pts[22], pts[26], pts[[22, 23, 24, 25, 26]])
    # 04 eyebrow_steepness_l
    features[4] = -np.rad2deg(a) - norm_angle
    # 05 eyebrow_quirk_l
    features[5] = np.max(np.abs(np.array(f_pts[1:4]) - f_pts[0, 1])) / norm_distance_y
    
    a, f_pts = align_points(pts[17], pts[21], pts[[17, 18, 19, 20, 21]])
    # 06 eyebrow_steepness_r
    features[6] = np.rad2deg(a) - norm_angle
    # 07 eyebrow_quirk_r
    features[7] = np.max(np.abs(np.array(f_pts[1:4]) - f_pts[0, 1])) / norm_distance_y

    # 08 eyebrow_down_l
    features[8] = (np.mean([pts[22, 1], pts[26, 1]]) - pts[27, 1]) / norm_distance_y
    # 09 eyebrow_down_r
    features[9] = f = (np.mean([pts[17, 1], pts[21, 1]]) - pts[27, 1]) / norm_distance_y

    upper_mouth_line = np.mean([pts[49, 1], pts[50, 1], pts[51, 1]])
    center_line = np.mean([pts[50, 0], pts[60, 0], pts[27, 0], pts[30, 0], pts[64, 0], pts[55, 0]])
    # 10 mouth_corner_down_l
    features[10] = (upper_mouth_line - pts[62, 1]) / norm_distance_y
    # 11 mouth_corner_down_r
    features[11] = (upper_mouth_line - pts[58, 1]) / norm_distance_y
    # 12 mouth_corner_inout_l
    features[12] = abs(center_line - pts[62, 0]) / norm_distance_x
    # 13 mouth_corner_inout_r
    features[13] = abs(center_line - pts[58, 0]) / norm_distance_x

    # 14 mouth_open
    features[14] = abs(np.mean(pts[[59,60,61], 1], axis=0) - np.mean(pts[[63,64,65], 1], axis=0)) / norm_distance_y

    # 15 mouth_wide
    features[15] = abs(pts[58, 0] - pts[62, 0]) / norm_distance_x

    return features

cdef public void run_osf(char *c_image, face_t *features_out):
    # image from C to numpy
    cdef char [:, :, :] image_view = <char [:HEIGHT, :WIDTH, :2]> c_image
    frame = cv2.cvtColor(np.ndarray((HEIGHT, WIDTH, 2), buffer=image_view, order='c', dtype=np.uint8), cv2.COLOR_YUV2RGB_YUYV)
    
    # preprocess for detection model
    im = np.transpose(
        np.expand_dims(
            cv2.resize(frame, (MODEL_RES, MODEL_RES), interpolation=cv2.INTER_LINEAR) * std_224 + mean_224,
            0
        ),
        (0,3,1,2)
    )

    # run detection model
    outputs, maxpool = detection_model.run([], {'input': im})

    # work out most confident box containing a face
    outputs = np.array(outputs)
    maxpool = np.array(maxpool)
    outputs[0, 0, outputs[0, 0] != maxpool[0, 0]] = 0
    detections = np.flip(np.argsort(outputs[0,0].flatten()))
    det = detections[0]
    y, x = det // 56, det % 56
    c = outputs[0, 0, y, x]
    r = outputs[0, 1, y, x] * 112.
    x *= 4
    y *= 4
    r *= 1.0
    if c < detection_threshold:
        features_out.track_fail = 1
        return
    face_box = np.array((x - r, y - r, 2 * r, 2 * r * 1.0)).astype(np.float32)
    face_box[[0,2]] *= WIDTH / 224.
    face_box[[1,3]] *= HEIGHT / 224.

    # preprocess for landmark model
    x,y,w,h = face_box
    crop_x1 = x - int(w * 0.1)
    crop_y1 = y - int(h * 0.125)
    crop_x2 = x + w + int(w * 0.1)
    crop_y2 = y + h + int(h * 0.125)
    crop_x1, crop_y1 = clamp_to_im((crop_x1, crop_y1), WIDTH, HEIGHT)
    crop_x2, crop_y2 = clamp_to_im((crop_x2, crop_y2), WIDTH, HEIGHT)
    scale_x = float(crop_x2 - crop_x1) / res
    scale_y = float(crop_y2 - crop_y1) / res
    crop_0 = np.transpose(
        np.expand_dims(
            cv2.resize(
                frame[crop_y1:crop_y2, crop_x1:crop_x2],
                (MODEL_RES, MODEL_RES),
                interpolation=cv2.INTER_LINEAR
            ) * std_224 + mean_224,
            0
        ),
        (0,3,1,2)
    )

    # run landmark model
    tensor = lm_model.run([], {lm_input_name: crop_0})[0][0]

    # work out the landmarks
    res_minus_1 = MODEL_RES - 1
    c0, c1, c2 = 66, 132, 198
    t_main = tensor[0:c0].reshape((c0,out_res_i * out_res_i))
    t_m = t_main.argmax(1)
    indices = np.expand_dims(t_m, 1)
    t_conf = np.take_along_axis(t_main, indices, 1).reshape((c0,))
    t_off_x = np.take_along_axis(tensor[c0:c1].reshape((c0,out_res_i * out_res_i)), indices, 1).reshape((c0,))
    t_off_y = np.take_along_axis(tensor[c1:c2].reshape((c0,out_res_i * out_res_i)), indices, 1).reshape((c0,))
    t_off_x = res_minus_1 * logit_arr(t_off_x, logit_factor)
    t_off_y = res_minus_1 * logit_arr(t_off_y, logit_factor)
    t_x = crop_y1 + scale_y * (res_minus_1 * np.floor(t_m / out_res_i) / out_res + t_off_x)
    t_y = crop_x1 + scale_x * (res_minus_1 * np.floor(np.mod(t_m, out_res_i)) / out_res + t_off_y)
    # avg_conf = np.average(t_conf)
    lms = np.stack([t_x, t_y, t_conf], 1)
    lms[np.isnan(lms).any(axis=1)] = np.array([0.,0.,0.], dtype=np.float32)

    # TODO: gaze tracking
    eye_state = [(1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)]

    # TODO: dynamic face_3d fitting
    face_info.contour = np.array(face_info.face_3d[face_info.contour_pts])

    # work out spatial orientation
    lms = np.concatenate((lms, np.array([[eye_state[0][1], eye_state[0][2], eye_state[0][3]], [eye_state[1][1], eye_state[1][2], eye_state[1][3]]], np.float32)), 0)
    image_pts = np.array(lms)[face_info.contour_pts, 0:2]

    # solve Perspective-n-Point problem
    success, rotation, translation = cv2.solvePnP(face_info.contour, image_pts, camera, dist_coeffs, useExtrinsicGuess=True, rvec=np.transpose(face_info.rotation), tvec=np.transpose(face_info.translation), flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        features_out.track_fail = 1
        return
    else:
        # store to accelerate future PnP solving
        face_info.rotation = np.transpose(rotation)
        face_info.translation = np.transpose(translation)

    # work out all landmarks
    pts_3d = np.zeros((70,3), np.float32)
    rmat, _ = cv2.Rodrigues(rotation)
    quaternion = matrix_to_quaternion(rmat)
    inverse_rotation = np.linalg.inv(rmat)
    t_reference = face_info.face_3d.dot(rmat.transpose())
    t_reference = t_reference + face_info.translation
    t_reference = t_reference.dot(camera.transpose())
    t_depth = t_reference[:, 2]
    t_depth[t_depth == 0] = 0.000001
    t_depth_e = np.expand_dims(t_depth[:],1)
    t_reference = t_reference[:] / t_depth_e
    pts_3d[0:66] = np.stack([lms[0:66,0], lms[0:66,1], np.ones((66,))], 1) * t_depth_e[0:66]
    pts_3d[0:66] = (pts_3d[0:66].dot(inverse_camera.transpose()) - face_info.translation).dot(inverse_rotation.transpose())
    for i, pt in enumerate(face_info.face_3d[66:70]):
        if i == 2:
            # Right eyeball
            # Eyeballs have an average diameter of 12.5mm and and the distance between eye corners is 30-35mm, so a conversion factor of 0.385 can be applied
            eye_center = (pts_3d[36] + pts_3d[39]) / 2.0
            d_corner = np.linalg.norm(pts_3d[36] - pts_3d[39])
            depth = 0.385 * d_corner
            pt_3d = np.array([eye_center[0], eye_center[1], eye_center[2] - depth])
            pts_3d[68] = pt_3d
            continue
        if i == 3:
            # Left eyeball
            eye_center = (pts_3d[42] + pts_3d[45]) / 2.0
            d_corner = np.linalg.norm(pts_3d[42] - pts_3d[45])
            depth = 0.385 * d_corner
            pt_3d = np.array([eye_center[0], eye_center[1], eye_center[2] - depth])
            pts_3d[69] = pt_3d
            continue
        if i == 0:
            d1 = np.linalg.norm(lms[66,0:2] - lms[36,0:2])
            d2 = np.linalg.norm(lms[66,0:2] - lms[39,0:2])
            d = d1 + d2
            pt = (pts_3d[36] * d1 + pts_3d[39] * d2) / d
        if i == 1:
            d1 = np.linalg.norm(lms[67,0:2] - lms[42,0:2])
            d2 = np.linalg.norm(lms[67,0:2] - lms[45,0:2])
            d = d1 + d2
            pt = (pts_3d[42] * d1 + pts_3d[45] * d2) / d
        if i < 2:
            reference = rmat.dot(pt)
            reference = reference + face_info.translation
            reference = camera.dot(reference)
            depth = reference[2]
            pt_3d = np.array([lms[66+i][0] * depth, lms[66+i][1] * depth, depth], np.float32)
            pt_3d = inverse_camera.dot(pt_3d)
            pt_3d = pt_3d - face_info.translation
            pt_3d = inverse_rotation.dot(pt_3d)
            pts_3d[66+i,:] = pt_3d[:]

    # normalize pts_3d
    pts_3d[np.isnan(pts_3d).any(axis=1)] = np.array([0.,0.,0.], dtype=np.float32)
    pts_3d = normalize_pts3d(pts_3d)

    # solve features
    features = solve_features(pts_3d[:, 0:2])

    # populate result struct
    features_out.track_fail = 0
    features_out.translation[0] = face_info.translation[0]
    features_out.translation[1] = face_info.translation[1]
    features_out.translation[2] = face_info.translation[2]
    features_out.rotation[0] = face_info.rotation[0]
    features_out.rotation[1] = face_info.rotation[1]
    features_out.rotation[2] = face_info.rotation[2]
    features_out.eye[0], features_out.eye[1] = features[0], features[1]
    features_out.eye_blink[0], features_out.eye_blink[1] = features[2], features[3]
    features_out.eyebrow_steepness[0], features_out.eyebrow_steepness[1] = features[4], features[6]
    features_out.eyebrow_quirk[0], features_out.eyebrow_quirk[1] = features[5], features[7]
    features_out.eyebrow_down[0], features_out.eyebrow_down[1] = features[8], features[9]
    features_out.mouth_corner_down[0], features_out.mouth_corner_down[1] = features[10], features[11]
    features_out.mouth_corner_inout[0], features_out.mouth_corner_inout[1] = features[12], features[13]
    features_out.mouth_open = features[14]
    features_out.mouth_wide = features[15]

    return

print("OpenSeeFace Lite loaded.")