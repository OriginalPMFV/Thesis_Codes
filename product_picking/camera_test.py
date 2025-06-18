#!/usr/bin/env python3
"""
Mostra profundidade e devolve (X,Y,Z) quando o rato passa sobre o pixel,
e aplica o modelo YOLOv11 para deteção de frutos em tempo real.
Requisitos:
    pip install pyrealsense2 opencv-python numpy ultralytics
Câmara testada: Intel RealSense D435i
"""

import numpy as np
import cv2 as cv
import pyrealsense2 as rs
import sys, math
from ultralytics import YOLO

# --------- Configuração da câmera e modelo ----------
pipeline = rs.pipeline()  # Cria pipeline para capturar dados da câmera
cfg = rs.config()  # Configura os streams
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Ativa stream de profundidade
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Ativa stream de cor

print("[INFO] A iniciar pipeline…")
profile = pipeline.start(cfg)  # Inicia a pipeline

align = rs.align(rs.stream.color)  # Alinha profundidade com imagem RGB

depth_intrin = profile.get_stream(rs.stream.depth) \
    .as_video_stream_profile().get_intrinsics()  # Obtém os intrínsecos da câmara de profundidade

# Carrega modelo YOLOv11 treinado
yolo_model = YOLO("C:/Users/Utilizador/PycharmProjects/TesteRTDE_novo/best.pt")
yolo_model.fuse()  # Otimiza o modelo para inferência
try:
    yolo_model.to('cuda:0').half()  # Usa GPU se disponível (meia precisão)
except Exception:
    pass  # Se não tiver GPU, continua com CPU

# ---------- Função utilitária de projeção ----------
def pixel_to_point(u, v, depth_value, intrin=depth_intrin):
    # Converte coordenadas de pixel (u,v) + profundidade → coordenadas 3D reais (X,Y,Z)
    if depth_value <= 0 or math.isnan(depth_value):
        return None
    X, Y, Z = rs.rs2_deproject_pixel_to_point(intrin, [u, v], depth_value)
    return np.array([X, Y, Z])

# ---------- Callback de mouse ----------
xyz_text = ""
def mouse_move(event, x, y, flags, param):
    global xyz_text
    if event == cv.EVENT_MOUSEMOVE:
        # Quando o rato se move, obtém a profundidade do pixel (x,y)
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        depth_frame = aligned.get_depth_frame()
        if not depth_frame:
            xyz_text = "Sem frame de depth"
            return
        d = depth_frame.get_distance(x, y)  # Distância em metros
        pt = pixel_to_point(x, y, d)  # Converte para coordenadas 3D
        if pt is not None:
            xyz_text = f"X:{pt[0]:.3f} Y:{pt[1]:.3f} Z:{pt[2]:.3f} m"
        else:
            xyz_text = "Sem profundidade válida"

# ---------- Janela principal ----------
cv.namedWindow("RealSense XYZ + YOLO", cv.WINDOW_AUTOSIZE)  # Cria janela de visualização
cv.setMouseCallback("RealSense XYZ + YOLO", mouse_move)  # Liga callback do rato

try:
    while True:
        # Captura e alinha os frames depth e color
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not color_frame or not depth_frame:
            continue

        img = np.asanyarray(color_frame.get_data())  # Converte para array OpenCV

        # Aplica YOLO ao frame de cor
        det = yolo_model(img,
                         verbose=False,
                         conf=0.5,
                         iou=0.45,
                         imgsz=640)[0]
        # Desenha bounding boxes das detecções
        for box, cls_id, conf in zip(det.boxes.xyxy, det.boxes.cls, det.boxes.conf):
            x1, y1, x2, y2 = map(int, box.tolist())
            cv.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv.putText(img, f"ID{int(cls_id)}:{conf:.2f}",
                       (x1, y1-5), cv.FONT_HERSHEY_SIMPLEX,
                       0.5, (255,255,255), 1)

        # Mostra texto com coordenadas XYZ do cursor
        cv.rectangle(img, (0,0), (300,25), (0,0,0), -1)
        cv.putText(img, xyz_text, (10,18),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        cv.imshow("RealSense XYZ + YOLO", img)  # Mostra imagem com deteções
        key = cv.waitKey(1)
        if key == 27:  # Tecla ESC para sair
            break

finally:
    pipeline.stop()  # Liberta recursos da câmera
    cv.destroyAllWindows()  # Fecha janelas
