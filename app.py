from flask import Flask, request, jsonify
from PIL import Image, ImageOps
import mediapipe as mp
import numpy as np
import io
import base64
import traceback

app = Flask(__name__)

# Inicializa detector facial do MediaPipe
mp_face_detection = mp.solutions.face_detection

face_detector = mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.60
)


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def center_crop_square(img):
    """
    Faz crop central quadrado sem deformar a imagem.
    Usado como fallback quando nenhum rosto é detectado.
    """
    width, height = img.size
    side = min(width, height)

    left = int((width - side) / 2)
    top = int((height - side) / 2)

    right = left + side
    bottom = top + side

    return img.crop((left, top, right, bottom))


def detect_main_face(image_np):
    """
    Detecta o rosto principal da imagem.
    Se houver mais de um rosto, escolhe o maior.
    """
    result = face_detector.process(image_np)

    if not result.detections:
        return None, 0.0

    best_detection = max(
        result.detections,
        key=lambda d: (
            d.location_data.relative_bounding_box.width *
            d.location_data.relative_bounding_box.height
        )
    )

    confidence = float(best_detection.score[0])
    bbox = best_detection.location_data.relative_bounding_box

    return bbox, confidence


def crop_face_smart(img, output_size=600):
    """
    Detecta o rosto e gera uma imagem quadrada, enquadrada e sem deformação.

    Regra:
    - corrige orientação EXIF;
    - detecta rosto;
    - calcula crop quadrado com margem;
    - garante que o crop fique dentro da imagem;
    - só depois redimensiona para output_size x output_size.
    """
    img = ImageOps.exif_transpose(img).convert("RGB")
    image_np = np.array(img)

    img_h, img_w = image_np.shape[:2]

    bbox, confidence = detect_main_face(image_np)

    # Fallback: se não detectar rosto, faz crop central quadrado
    if bbox is None:
        cropped = center_crop_square(img)
        cropped = cropped.resize((output_size, output_size), Image.LANCZOS)

        debug_info = {
            "mode": "center_crop_fallback",
            "image_width": img_w,
            "image_height": img_h,
            "face_detected": False,
            "confidence": confidence
        }

        return cropped, False, confidence, debug_info

    # Bounding box relativo -> absoluto
    x = int(bbox.xmin * img_w)
    y = int(bbox.ymin * img_h)
    w = int(bbox.width * img_w)
    h = int(bbox.height * img_h)

    # Corrige possíveis valores negativos vindos do detector
    x = clamp(x, 0, img_w - 1)
    y = clamp(y, 0, img_h - 1)
    w = clamp(w, 1, img_w - x)
    h = clamp(h, 1, img_h - y)

    # Centro horizontal do rosto
    cx = x + (w / 2)

    # Centro vertical levemente deslocado para baixo
    # para incluir boca, queixo, pescoço e um pouco de ombro.
    cy = y + (h * 0.62)

    # Tamanho do crop quadrado.
    # Aumente esses multiplicadores se quiser mais corpo/fundo.
    side = int(max(w * 2.5, h * 3.0))

    # Garante que o crop quadrado caiba na imagem.
    # Isso é o ponto crítico para não deformar no resize.
    side = min(side, img_w, img_h)

    # Calcula posição inicial.
    # O top usa 0.45 para deixar um pouco menos espaço acima e mais abaixo.
    left = int(round(cx - side / 2))
    top = int(round(cy - side * 0.45))

    # Ajusta para manter o quadrado totalmente dentro dos limites.
    left = clamp(left, 0, img_w - side)
    top = clamp(top, 0, img_h - side)

    right = left + side
    bottom = top + side

    # Segurança: garante crop quadrado real
    crop_width = right - left
    crop_height = bottom - top

    if crop_width != crop_height:
        # fallback defensivo; não deveria ocorrer
        side = min(crop_width, crop_height)
        right = left + side
        bottom = top + side

    cropped = img.crop((left, top, right, bottom))
    cropped = cropped.resize((output_size, output_size), Image.LANCZOS)

    debug_info = {
        "mode": "face_crop",
        "image_width": img_w,
        "image_height": img_h,
        "face_detected": True,
        "confidence": confidence,
        "face_box": {
            "x": x,
            "y": y,
            "width": w,
            "height": h
        },
        "crop_box": {
            "left": left,
            "top": top,
            "right": right,
            "bottom": bottom,
            "side": side
        },
        "output_size": output_size
    }

    return cropped, True, confidence, debug_info


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "success": True,
        "message": "API ativa"
    }), 200


@app.route("/auto-crop-face", methods=["POST"])
def auto_crop_face():
    """
    Endpoint principal.

    Espera:
      multipart/form-data
      campo: image
      campo opcional: output_size

    Exemplo:
      curl.exe -X POST "http://127.0.0.1:5000/auto-crop-face" `
        -F "image=@C:\\Teste\\barth.png" `
        -F "output_size=600"
    """
    if "image" not in request.files:
        return jsonify({
            "success": False,
            "message": "Campo 'image' não enviado."
        }), 400

    file = request.files["image"]

    if not file or file.filename == "":
        return jsonify({
            "success": False,
            "message": "Arquivo vazio."
        }), 400

    try:
        output_size_raw = request.form.get("output_size", "600")

        try:
            output_size = int(output_size_raw)
        except ValueError:
            return jsonify({
                "success": False,
                "message": "output_size deve ser um número inteiro."
            }), 400

        if output_size < 100 or output_size > 2000:
            return jsonify({
                "success": False,
                "message": "output_size deve estar entre 100 e 2000."
            }), 400

        img = Image.open(file.stream)

        cropped_img, face_detected, confidence, debug_info = crop_face_smart(
            img,
            output_size=output_size
        )

        buffer = io.BytesIO()
        cropped_img.save(buffer, format="JPEG", quality=90, optimize=True)

        image_bytes = buffer.getvalue()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        return jsonify({
            "success": True,
            "face_detected": face_detected,
            "confidence": round(confidence, 4),
            "mime_type": "image/jpeg",
            "filename": "foto_perfil.jpg",
            "width": output_size,
            "height": output_size,
            "image_base64": image_base64,
            "debug": debug_info
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Erro ao processar imagem: {str(e)}",
            "trace": traceback.format_exc()
        }), 500


@app.route("/auto-crop-face-file", methods=["POST"])
def auto_crop_face_file():
    """
    Endpoint alternativo: retorna a imagem diretamente como arquivo JPEG.
    Útil para teste visual rápido no Postman/browser.
    Para integração com APEX via JSON, use /auto-crop-face.
    """
    if "image" not in request.files:
        return jsonify({
            "success": False,
            "message": "Campo 'image' não enviado."
        }), 400

    file = request.files["image"]

    if not file or file.filename == "":
        return jsonify({
            "success": False,
            "message": "Arquivo vazio."
        }), 400

    try:
        output_size_raw = request.form.get("output_size", "600")

        try:
            output_size = int(output_size_raw)
        except ValueError:
            return jsonify({
                "success": False,
                "message": "output_size deve ser um número inteiro."
            }), 400

        img = Image.open(file.stream)

        cropped_img, face_detected, confidence, debug_info = crop_face_smart(
            img,
            output_size=output_size
        )

        buffer = io.BytesIO()
        cropped_img.save(buffer, format="JPEG", quality=90, optimize=True)
        buffer.seek(0)

        from flask import send_file

        return send_file(
            buffer,
            mimetype="image/jpeg",
            as_attachment=False,
            download_name="foto_perfil.jpg"
        )

    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Erro ao processar imagem: {str(e)}",
            "trace": traceback.format_exc()
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )