import easyocr
import supervision as sv
import cv2
import numpy as np
import pandas as pd # <-- Importamos pandas para trabajar con Excel

# --- Configuración ---
Image_path = 'soloimpresion.png' # <--- CAMBIA ESTO AL NOMBRE DE TU ARCHIVO DE IMAGEN
model_storage_directory = '.'

# Inicializa el lector de EasyOCR.
reader = easyocr.Reader(['es'], gpu=False, model_storage_directory=model_storage_directory)

# Realiza la detección de texto en la imagen.
result = reader.readtext(Image_path)

# Carga la imagen con OpenCV.
image = cv2.imread(Image_path)

# --- Manejo de errores al cargar la imagen ---
if image is None:
    print(f"Error: No se pudo cargar la imagen desde '{Image_path}'")
    print("Por favor, verifica que el archivo exista y que la ruta sea correcta.")
    exit()

# Prepara listas para los datos de las detecciones (para la anotación y para el Excel).
xyxy, confidences, class_ids, labels = [], [], [], []

# Lista para almacenar los datos que irán al Excel
excel_data = []

# Extrae los datos del OCR y los prepara para Supervision y para Excel
for detection in result:
    bbox, text, confidence = detection[0], detection[1], detection[2]

    x_min = int(min([point[0] for point in bbox]))
    y_min = int(min([point[1] for point in bbox]))
    x_max = int(max([point[0] for point in bbox]))
    y_max = int(max([point[1] for point in bbox]))

    # Agrega los datos para Supervision
    xyxy.append([x_min, y_min, x_max, y_max])
    labels.append(text)
    confidences.append(confidence)
    class_ids.append(0)

    # Agrega los datos para el Excel
    excel_data.append({
        'Texto Detectado': text,
        'Confianza': round(confidence, 4), # Redondeamos la confianza para mayor claridad
        'Coordenada X Mínima': x_min,
        'Coordenada Y Mínima': y_min,
        'Coordenada X Máxima': x_max,
        'Coordenada Y Máxima': y_max
    })

# --- Guardar los datos en un archivo de Excel ---
if excel_data: # Solo guarda si hay datos para evitar crear un archivo vacío
    df = pd.DataFrame(excel_data) # Crea un DataFrame de pandas con los datos
    excel_output_path = "easyocr_detecciones.xlsx"
    df.to_excel(excel_output_path, index=False) # Guarda el DataFrame en un archivo Excel
    print(f"Datos guardados exitosamente en '{excel_output_path}'")
else:
    print("No se detectó texto en la imagen. No se creó ningún archivo de Excel.")


# Crea un objeto `Detections` de Supervision.
detections = sv.Detections(
    xyxy=np.array(xyxy),
    confidence=np.array(confidences),
    class_id=np.array(class_ids)
)

# Inicializa los objetos anotadores de Supervision.
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Anota la imagen.
annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image,
    detections=detections,
    labels=labels
)

# --- AJUSTE PARA MOSTRAR LA IMAGEN COMPLETA EN PANTALLA ---
(h, w) = annotated_image.shape[:2]
max_display_width = 1200
max_display_height = 800

if w > max_display_width or h > max_display_height:
    width_scale = max_display_width / w
    height_scale = max_display_height / h
    scale = min(width_scale, height_scale)

    new_width = int(w * scale)
    new_height = int(h * scale)

    display_image = cv2.resize(annotated_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
else:
    display_image = annotated_image.copy()

# --- Mostrar y Guardar la Imagen Anotada (para Entornos Locales) ---
cv2.imshow('Imagen Anotada', display_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("Output.jpg", annotated_image) # Guarda la imagen anotada en su tamaño original

print("Procesamiento completado. La imagen anotada se ha guardado como Output.jpg")