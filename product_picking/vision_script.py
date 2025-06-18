#!/usr/bin/env python3
#  detect_yolov11_to_tcp.py
#
#  Detecta frutos com YOLOv11 numa Intel RealSense.
#  • Durante 5 s faz inferência contínua (≈150 frames); apenas o
#    ÚLTIMO frame é mostrado e usado para devolver:
#        (classe, X, Y, Z, width)  ← width = largura real do fruto [m]
# ---------------------------------------------------------------------------

import time
import cv2 as cv
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

from control_script import principal   # chamada original

# ────────── parâmetros ─────────────────────────────────────────
MODEL_PATH   = "C:/Users/Utilizador/PycharmProjects/TesteRTDE_novo/best.pt"
HSV_LOWER    = (0, 90, 40)                       # faixa HSV da cor-alvo
HSV_UPPER    = (10, 255, 255)
WARM_UP_SEC  = 5
FPS          = 30                                # assumed RealSense FPS

model: YOLO | None = None


# ────────── auxiliares ─────────────────────────────────────────
def pixel_to_cam(u: int, v: int, depth_m: float, intr: rs.intrinsics) -> np.ndarray:
    """pixel (u,v) + profundidade [m] → ponto 3-D na câmara [m]"""
    X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [u, v], depth_m)
    return np.array([X, Y, Z])


def hsv_pass(roi: np.ndarray) -> bool:
    """Retorna True se a ROI contiver a cor-alvo"""
    hsv  = cv.cvtColor(roi, cv.COLOR_BGR2HSV)             # converte ROI para HSV
    mask = cv.inRange(hsv, HSV_LOWER, HSV_UPPER)          # aplica máscara da cor desejada
    return cv.countNonZero(mask) > 0                      # retorna se há pixels válidos


# ────────── detecção ───────────────────────────────────────────
def detect_once() -> list[tuple[int, float, float, float, float]]:
    """
    Inferência contínua durante 5 s; devolve lista:
        (classe, X, Y, Z, largura)  em metros,
    calculada no ÚLTIMO frame captado.
    """

    global model

    # RealSense
    pipe = rs.pipeline()
    cfg  = rs.config()
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, FPS)    # stream de cor
    cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, FPS)     # stream de profundidade
    pipe.start(cfg)
    align = rs.align(rs.stream.color)                                     # alinhamento depth → color

    # carrega o modelo se ainda não estiver carregado
    if model is None:
        print("🔄  A carregar YOLOv11…", flush=True)
        model = YOLO(MODEL_PATH)
        print("✅  Modelo pronto.", flush=True)

    # guarda o último frame e deteções
    last_img   = last_depth = last_intr = last_det = None

    t0 = time.time()
    while time.time() - t0 < WARM_UP_SEC:
        frames          = align.process(pipe.wait_for_frames())
        color, depth_fr = frames.get_color_frame(), frames.get_depth_frame()
        if not (color and depth_fr):
            continue

        img   = np.asanyarray(color.get_data())                               # imagem BGR
        depth = np.asanyarray(depth_fr.get_data(), dtype=np.float32) * 0.001  # profundidade em metros
        intr  = color.profile.as_video_stream_profile().intrinsics            # intrínsecos da câmara

        det   = model(img, verbose=False, conf=0.8)[0]                         # deteções do YOLO

        last_img, last_depth, last_intr, last_det = img, depth, intr, det     # guarda último frame

    if last_img is None:
        pipe.stop()
        raise RuntimeError("Não foi possível capturar frame durante o warm-up")

    # ─── processa o ÚLTIMO frame ───
    fruits_tcp: list[tuple[int, float, float, float, float]] = []

    for box, cid, conf in zip(last_det.boxes.xyxy,
                              last_det.boxes.cls,
                              last_det.boxes.conf):
        x1, y1, x2, y2 = map(int, box.tolist())              # coordenadas da bbox
        u, v           = (x1 + x2) // 2, (y1 + y2) // 2       # centro do bounding box

        # média da profundidade 3x3 no centro da bbox
        z_patch = last_depth[max(0, v - 1):v + 2, max(0, u - 1):u + 2]
        valid   = z_patch[z_patch > 0]
        if valid.size == 0:
            continue
        z = float(valid.mean())

        # filtro de cor HSV dentro da bounding box
        if not hsv_pass(last_img[y1:y2, x1:x2]):
            continue

        # coordenadas 3D reais
        p_cam = pixel_to_cam(u, v, z, last_intr)

        # -------- largura real do fruto --------
        width_px = x2 - x1                                  # largura em pixeis
        width_m  = width_px * z / last_intr.fx              # largura em metros

        fruits_tcp.append((int(cid), *p_cam, width_m))      # adiciona à lista final

        # visualização (opcional)
        cv.rectangle(last_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv.putText(last_img,
                   f"{int(cid)}:{conf:.2f}",
                   (x1, y1 - 5),
                   cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv.putText(last_img,
                   f"{width_m*1000:.0f} mm",
                   (x1, y2 + 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # mostra o último frame processado com deteções
    cv.imshow("YOLOv11 detections (último frame do warm-up)", last_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    pipe.stop()

    return fruits_tcp


# ────────── execução principal (teste) ─────────────────────────
if __name__ == "__main__":
    for i in range(5):                          # executa 5 ciclos de deteção
        tic  = time.time()
        lst  = detect_once()                    # deteções: (classe, X, Y, Z, largura)
        print("Detecções:")
        for cid, x, y, z, w in lst:
            print(f"ID {cid}  Xcam={x:+.3f}  Ycam={y:+.3f}  Zcam={z:+.3f}  Width={w*1000:.0f} mm")
        print(f"Elapsed {time.time() - tic:.2f}s\n")

        principal(lst)                          # envia lista de deteções para controlo do robô
