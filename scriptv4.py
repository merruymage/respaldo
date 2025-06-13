import easyocr
import supervision as sv
import cv2
import numpy as np
import pandas as pd
from sympy import python

# --- Configuración ---
Image_path = '27084609001.jpg' # Asegúrate de que esta ruta sea correcta
model_storage_directory = '.' # Directorio para los modelos de EasyOCR

# --- TUS COORDENADAS DE LAS REGIONES DE INTERÉS (IMAGEN ORIGINAL) ---
# Cada elemento de la lista es una ROI: [x_min, y_min, x_max, y_max]
regions_of_interest = [
    [289, 889, 948, 920],   # ROI 1
    [965, 881, 1672, 920],  # ROI 2
    [1817, 854, 2181, 894], # ROI 3
    [316, 981, 820, 1056],  # ROI 4
    [1483, 973, 1852, 1025],# ROI 5
    [1878, 968, 2190, 1017],# ROI 6
    [513, 1091, 1759, 1139],# ROI 7
    [272, 1201, 680, 1249], # ROI 8
    [706, 1196, 1119, 1240],# ROI 9
    [1145, 1192, 1597, 1240],# ROI 10
    [1628, 1187, 2181, 1231],# ROI 11
    [1435, 1293, 2163, 1337],# ROI 12
    [1584, 1394, 2181, 1451],# ROI 13
    [1084, 1402, 1553, 1446],# ROI 14
    [592, 1455, 1215, 1543],# ROI 15
    [583, 1613, 1215, 1665],# ROI 16
    [1571, 1608, 2159, 1661],# ROI 17
    [280, 1810, 1062, 1863],# ROI 18
    [276, 1871, 1057, 1928],# ROI 19
    [276, 1942, 1070, 1994],# ROI 20
    [267, 2003, 1057, 2060],# ROI 21
    [267, 2069, 1057, 2126],# ROI 22
    [1145, 1797, 1483, 1854],# ROI 23
    [1145, 1867, 1492, 1920],# ROI 24
    [1145, 1928, 1496, 1985],# ROI 25
    [1145, 1990, 1496, 2051],# ROI 26
    [1136, 2064, 1487, 2112],# ROI 27
    [1514, 1792, 1891, 1854],# ROI 28
    [1518, 1867, 1891, 1928],# ROI 29
    [1522, 1928, 1896, 1981],# ROI 30
    [1527, 1994, 1900, 2055],# ROI 31
    [1531, 2060, 1891, 2126],# ROI 32
    [2036, 1784, 2207, 1836],# ROI 33
    [2045, 1854, 2203, 1920],# ROI 34
    [2032, 1911, 2198, 1981],# ROI 35
    [2032, 1977, 2194, 2038],# ROI 36
    [2040, 2042, 2190, 2104],# ROI 37
    [258, 2612, 741, 2704]  # ROI 38
]

# Inicializa el lector de EasyOCR.
reader = easyocr.Reader(['es'], gpu=False, model_storage_directory=model_storage_directory)

# Carga la imagen con OpenCV.
image = cv2.imread(Image_path)

# --- Manejo de errores al cargar la imagen ---
if image is None:
    print(f"Error: No se pudo cargar la imagen desde '{Image_path}'")
    print("Por favor, verifica que el archivo exista y que la ruta sea correcta.")
    exit()



### Image Preprocessing for Handwriting



# --- PREPROCESAMIENTO DE IMAGEN PARA MEJORAR LA LECTURA DE CALIGRAFÍA ---
# 1. Convertir a escala de grises
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 2. Eliminar ruido usando Desenfoque Gaussiano
# Kernel size (e.g., (5, 5)) depends on the noise level. Larger kernel for more blur.
# Use odd numbers for kernel dimensions.
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# 3. Binarización adaptativa: Esencial para caligrafía con iluminación variable
# cv2.ADAPTIVE_THRESH_GAUSSIAN_C es generalmente mejor para texto sobre fondo variable.
# cv2.THRESH_BINARY_INV invierte los colores (texto oscuro sobre fondo claro).
# Block size: Vecindario de píxeles para calcular el umbral (debe ser impar, ej., 11, 15, 21, 25).
#             Experimenta con este valor.
# C: Constante que se resta del umbral medio. Ajusta para hacer el texto más grueso o delgado.
binary_image = cv2.adaptiveThreshold(blurred_image, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                     25, 10) # <<-- AJUSTA ESTOS VALORES (25 y 10)

# 4. Operaciones Morfológicas para limpiar y conectar caracteres (Opcional, pero muy útil)
# Define un kernel pequeño para las operaciones.
kernel = np.ones((2,2),np.uint8) # Un kernel de 2x2 o 3x3 es común para letras.

# Opening: Erosión seguida de dilatación. Útil para eliminar pequeños puntos de ruido.
# Puedes aumentar 'iterations' para un efecto más fuerte.
processed_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=1)

# Closing: Dilatación seguida de erosión. Útil para cerrar pequeños huecos en las letras
# y conectar partes de caracteres rotos.
processed_image = cv2.morphologyEx(processed_image, cv2.MORPH_CLOSE, kernel, iterations=1)

# Opcional: Detección y eliminación de bordes gruesos si los hay (útil en algunos formularios)
# from skimage.filters import threshold_local
# try:
#     from deskew import determine_skew
# except ImportError:
#     print("Instala 'deskew' para corrección de inclinación: pip install deskew")

# --- PASAMOS LA IMAGEN PREPROCESADA A EASYOCR ---
# EasyOCR puede trabajar directamente con arrays de NumPy (la imagen de OpenCV es un array)
result = reader.readtext(processed_image, paragraph=False) # <--- ¡CAMBIO CLAVE AQUÍ! Pasamos la imagen procesada y usamos paragraph=False para mejor manejo de líneas.



### Post-processing and Data Handling

# Prepara listas para los datos de las detecciones que sí están dentro de las ROIs
filtered_xyxy, filtered_confidences, filtered_class_ids, filtered_labels = [], [], [], []
excel_data = []

# --- FILTRADO DE DETECCIONES BASADO EN LAS REGIONES DE INTERÉS ---
# Función auxiliar para verificar si un punto está dentro de un rectángulo
def is_point_inside_roi(point, roi):
    x, y = point
    roi_xmin, roi_ymin, roi_xmax, roi_ymax = roi
    return roi_xmin <= x <= roi_xmax and roi_ymin <= y <= roi_ymax

# Extrae los datos del OCR y los filtra por tus ROIs
for detection in result:
    bbox, text, confidence = detection[0], detection[1], detection[2]

    # Convertir bounding box format (EasyOCR da 4 puntos, Supervision espera min/max corners)
    # y también obtener el centroide del bounding box para una verificación sencilla
    x_min = int(min([point[0] for point in bbox]))
    y_min = int(min([point[1] for point in bbox]))
    x_max = int(max([point[0] for point in bbox]))
    y_max = int(max([point[1] for point in bbox]))

    # Calcula el centroide del bounding box
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2

    # Bandera para saber si la detección está en alguna de las ROIs
    is_in_roi = False
    for roi in regions_of_interest:
        # Usamos la lógica del centroide: si el centro del texto detectado está dentro del ROI.
        if is_point_inside_roi((center_x, center_y), roi):
            is_in_roi = True
            break
        # Alternativa: Si cualquier parte del bounding box del texto se solapa con el ROI.
        # This is more complex (intersection of two rectangles) but can be more inclusive.
        # For simplicity, sticking with centroid or corner checks is often enough.

    if is_in_roi:
        # Agrega los datos solo si la detección está dentro de alguna ROI
        filtered_xyxy.append([x_min, y_min, x_max, y_max])
        filtered_labels.append(text)
        filtered_confidences.append(confidence)
        filtered_class_ids.append(0) # Asigna un ID de clase predeterminado (e.g., 0 for 'text')

        # Agrega los datos para el Excel
        excel_data.append({
            'Texto Detectado': text,
            'Confianza': round(confidence, 4),
            'Coordenada X Mínima': x_min,
            'Coordenada Y Mínima': y_min,
            'Coordenada X Máxima': x_max,
            'Coordenada Y Máxima': y_max
        })

# --- Guardar los datos en un archivo de Excel ---
if excel_data:
    df = pd.DataFrame(excel_data)
    excel_output_path = "easyocr_detecciones.xlsx"
    df.to_excel(excel_output_path, index=False)
    print(f"Datos guardados exitosamente en '{excel_output_path}'")
else:
    print("No se detectó texto en la imagen dentro de las regiones especificadas. No se creó ningún archivo de Excel.")


# Crea un objeto `Detections` de Supervision a partir de los datos FILTRADOS
detections = sv.Detections(
    xyxy=np.array(filtered_xyxy),
    confidence=np.array(filtered_confidences),
    class_id=np.array(filtered_class_ids)
)

# Inicializa los objetos anotadores de Supervision.
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Anota la imagen con los cuadros delimitadores y etiquetas FILTRADOS
annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image,
    detections=detections,
    labels=filtered_labels # Pasamos las etiquetas filtradas
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

cv2.imwrite("Output.jpg", annotated_image)

print("Procesamiento completado. La imagen anotada se ha guardado como Output.jpg")