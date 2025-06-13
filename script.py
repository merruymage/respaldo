import easyocr       # Importa la librería EasyOCR para la detección de texto.
import supervision as sv # Importa la librería Supervision para la anotación de imágenes, como 'sv'.
import cv2           # Importa la librería OpenCV para el procesamiento de imágenes (leer, mostrar, guardar).
import numpy as np   # Importa NumPy para trabajar con arrays numéricos, que son la base de las imágenes en OpenCV.

# --- IMPORTANTE: Se ha eliminado la importación específica de Google Colab ---
# from google.colab.patches import cv2_imshow # ¡ESTA LÍNEA NO ES NECESARIA NI FUNCIONA FUERA DE GOOGLE COLAB!

# --- Configuración ---
# 1. Establece la RUTA REAL a tu archivo de imagen.
#    Ejemplo: 'C:/Users/TuUsuario/Pictures/mi_imagen.jpg'
#    Si la imagen está en la misma carpeta que este script, solo usa su nombre:
Image_path = 'soloimpresion.png' # "C:\Users\mbarreto\Desktop\prueba\10418477002.jpg" <--- CAMBIA ESTO AL NOMBRE DE TU ARCHIVO DE IMAGEN

#Kike no lo lee? miren a este: FUNCIONA TE DIJE!!!!

# 2. Establece un directorio donde EasyOCR pueda guardar sus modelos de lenguaje.
#    Usar '.' hará que los guarde en el directorio actual donde se ejecuta tu script.
model_storage_directory = '.'

# Inicializa el lector de EasyOCR.
# Se configura para detectar texto en español (['es']).
# 'gpu=False' fuerza el uso de la CPU, ideal si no tienes una GPU compatible.
# 'model_storage_directory' indica dónde guardar los modelos que EasyOCR descargue.
reader = easyocr.Reader(['es'], gpu=False, model_storage_directory=model_storage_directory)

# Realiza la detección de texto en la imagen especificada por Image_path.
result = reader.readtext(Image_path)

# Carga la imagen usando OpenCV.
image = cv2.imread(Image_path)

# --- Manejo de errores al cargar la imagen ---
# Verifica si la imagen se cargó correctamente. Si 'image' es None, significa que hubo un problema.
if image is None:
    print(f"Error: No se pudo cargar la imagen desde '{Image_path}'")
    print("Por favor, verifica que el archivo exista y que la ruta sea correcta.")
    exit() # Sale del script si la imagen no se puede cargar.

# Prepara listas vacías para almacenar las coordenadas de los cuadros delimitadores (bounding boxes),
# las confianzas de detección, los IDs de clase (genéricos para texto) y las etiquetas (el texto detectado).
xyxy, confidences, class_ids, labels = [], [], [], [] # Se renombró 'label' a 'labels' para mayor claridad.

# Extrae los datos del resultado de EasyOCR y los prepara para Supervision.
for detection in result:
    bbox, text, confidence = detection[0], detection[1], detection[2]

    # Convierte el formato del cuadro delimitador de EasyOCR (4 puntos de las esquinas)
    # al formato x_min, y_min, x_max, y_max que usa Supervision.
    x_min = int(min([point[0] for point in bbox]))
    y_min = int(min([point[1] for point in bbox]))
    x_max = int(max([point[0] for point in bbox]))
    y_max = int(max([point[1] for point in bbox]))

    # Agrega los datos procesados a sus respectivas listas.
    xyxy.append([x_min, y_min, x_max, y_max])
    labels.append(text)      # Usamos la lista 'labels' para el texto.
    confidences.append(confidence)
    class_ids.append(0)      # Asigna un ID de clase predeterminado (por ejemplo, 0 para "texto").

# Crea un objeto `Detections` de Supervision a partir de los arrays de NumPy.
# Este objeto encapsula todas las detecciones de manera estructurada.
detections = sv.Detections(
    xyxy=np.array(xyxy),
    confidence=np.array(confidences),
    class_id=np.array(class_ids)
)

# Inicializa los objetos anotadores de Supervision.
# `BoxAnnotator` para dibujar los cuadros delimitadores.
# `LabelAnnotator` para dibujar el texto de las etiquetas.
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Anota la imagen con los cuadros delimitadores y las etiquetas.
# Se usa 'image.copy()' para evitar modificar el array original de la imagen.
# `annotated_image` contendrá la imagen final con todas las anotaciones dibujadas.
annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image,
    detections=detections,
    labels=labels # Pasa la lista de textos detectados para que sean dibujados.
)

# --- Mostrar y Guardar la Imagen Anotada (para Entornos Locales como VS Code) ---

# sv.plot_image(image=annotated_image) # Esta línea es mejor para notebooks (Jupyter, Colab),
                                     # para entornos locales usamos cv2.imshow.

# 1. Muestra la imagen anotada en una nueva ventana emergente.
#    'Imagen Anotada' será el título de la ventana.
cv2.imshow('Imagen Anotada', annotated_image)

# 2. Espera indefinidamente hasta que se presione una tecla.
#    ¡Esto es crucial! Sin esto, la ventana aparecería y se cerraría al instante.
cv2.waitKey(0)

# 3. Cierra todas las ventanas de OpenCV que fueron abiertas por cv2.imshow().
cv2.destroyAllWindows()

# Crea la ventana con la bandera de redimensionable
cv2.namedWindow('Annotated Image', cv2.WINDOW_NORMAL) # <--- Cambia de cv2.WINDOW_AUTOSIZE a cv2.WINDOW_NORMAL

# Guarda la imagen anotada en un archivo llamado "Output.jpg" en el mismo directorio.
cv2.imwrite("Output.jpg", annotated_image)

# Imprime un mensaje en la consola indicando que el procesamiento ha finalizado.
print("Procesamiento completado. La imagen anotada se ha guardado como Output.jpg")

