import cv2

# Variables globales para almacenar las coordenadas del rectángulo
# y el estado de recorte.
ref_point = []  # Almacenará los puntos de inicio y fin del arrastre.
cropping = False # True si el usuario está actualmente arrastrando el ratón para recortar.
current_image_to_display = None # Almacenará la imagen que se muestra actualmente en la ventana.
original_image_full_size = None # Guardará la imagen original sin redimensionar.

# --- Callback de Ratón ---
def click_and_crop(event, x, y, flags, param):
    global ref_point, cropping, current_image_to_display, original_image_full_size

    # Cuando se presiona el botón izquierdo del ratón
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)] # Guarda el punto de inicio del arrastre.
        cropping = True      # Indica que estamos en modo de recorte.

    # Cuando el ratón se mueve mientras el botón izquierdo está presionado
    elif event == cv2.EVENT_MOUSEMOVE and cropping:
        # Crea una copia de la imagen actualmente visible (la redimensionada)
        temp_draw_image = current_image_to_display.copy()
        # Dibuja el rectángulo temporal desde el punto de inicio hasta la posición actual del ratón
        cv2.rectangle(temp_draw_image, ref_point[0], (x, y), (0, 255, 0), 2)
        # Muestra esta imagen temporal. Esto solo se hace aquí para mostrar el rectángulo en tiempo real.
        cv2.imshow("Selecciona la Region de Interes (Presiona 'q' para salir)", temp_draw_image)

    # Cuando se suelta el botón izquierdo del ratón
    elif event == cv2.EVENT_LBUTTONUP:
        ref_point.append((x, y)) # Guarda el punto final del arrastre.
        cropping = False         # Sale del modo de recorte.

        # Asegúrate de que los puntos estén ordenados para x_min, y_min, x_max, y_max
        x1, y1 = ref_point[0]
        x2, y2 = ref_point[1]
        x_min = min(x1, x2)
        y_min = min(y1, y2)
        x_max = max(x1, x2)
        y_max = max(y1, y2)

        # Muestra el rectángulo final en la imagen visible y original si se desea.
        # En este punto, puedes decidir si quieres dibujar el ROI permanente en la imagen que se muestra
        # o solo capturar las coordenadas. Aquí lo dibujamos para confirmación visual.
        cv2.rectangle(current_image_to_display, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.imshow("Selecciona la Region de Interes (Presiona 'q' para salir)", current_image_to_display)

        # Imprime las coordenadas de la ROI seleccionada.
        # Estas son las coordenadas de la imagen REDIMENSIONADA.
        print(f"ROI seleccionada (en la imagen mostrada): [x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}]")

        # --- CONVERSIÓN DE COORDENADAS A LA IMAGEN ORIGINAL DE TAMAÑO COMPLETO ---
        # Si la imagen fue redimensionada para la visualización,
        # necesitamos convertir las coordenadas seleccionadas de nuevo al tamaño original.
        if original_image_full_size is not None and current_image_to_display is not None:
            original_h, original_w = original_image_full_size.shape[:2]
            display_h, display_w = current_image_to_display.shape[:2]

            # Calcula los factores de escala inversos
            scale_x_inv = original_w / display_w
            scale_y_inv = original_h / display_h

            x_min_original = int(x_min * scale_x_inv)
            y_min_original = int(y_min * scale_y_inv)
            x_max_original = int(x_max * scale_x_inv)
            y_max_original = int(y_max * scale_y_inv)

            print(f"ROI seleccionada (en la imagen ORIGINAL): [x_min={x_min_original}, y_min={y_min_original}, x_max={x_max_original}, y_max={y_max_original}]")

            # Aquí puedes almacenar estas coordenadas originales en una lista
            # para pasarlas a EasyOCR más tarde.
            # Por ejemplo:
            # global selected_rois_for_ocr
            # selected_rois_for_ocr.append([x_min_original, y_min_original, x_max_original, y_max_original])

# --- Función para obtener el ROI interactivamente ---
def get_roi_interactively(image_path):
    global current_image_to_display, original_image_full_size

    # Carga la imagen original a tamaño completo
    original_image_full_size = cv2.imread(image_path)

    if original_image_full_size is None:
        print(f"Error: No se pudo cargar la imagen desde {image_path}")
        return []

    # --- Lógica de redimensionamiento para visualización ---
    (h, w) = original_image_full_size.shape[:2]
    max_display_width = 1200  # Ancho máximo deseado para la visualización
    max_display_height = 800  # Alto máximo deseado para la visualización

    if w > max_display_width or h > max_display_height:
        width_scale = max_display_width / w
        height_scale = max_display_height / h
        scale = min(width_scale, height_scale)

        new_width = int(w * scale)
        new_height = int(h * scale)

        current_image_to_display = cv2.resize(original_image_full_size.copy(), (new_width, new_height), interpolation=cv2.INTER_AREA)
    else:
        current_image_to_display = original_image_full_size.copy()

    # Crea la ventana y asigna el callback del ratón
    cv2.namedWindow("Selecciona la Region de Interes (Presiona 'q' para salir)")
    cv2.setMouseCallback("Selecciona la Region de Interes (Presiona 'q' para salir)", click_and_crop)

    print("\n--- Modo de Selección de ROI ---")
    print("Haz clic y arrastra para seleccionar una región.")
    print("Suelta el clic para ver las coordenadas.")
    print("Puedes seleccionar múltiples regiones. Las coordenadas se imprimirán.")
    print("Presiona 'q' en la ventana de la imagen para salir del modo de selección.")

    # Bucle principal para mostrar la imagen y capturar eventos de teclado
    while True:
        # Solo repintamos la imagen si no estamos en medio de un arrastre,
        # para evitar el temblor.
        if not cropping:
             cv2.imshow("Selecciona la Region de Interes (Presiona 'q' para salir)", current_image_to_display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"): # Presiona 'q' para salir
            break

    cv2.destroyAllWindows()
    return [] # Aquí deberías devolver la lista de ROIs seleccionadas si las almacenaste

# --- TU CÓDIGO PRINCIPAL INTEGRANDO LA SELECCIÓN DE ROI ---
# Asegúrate de que tu imagen esté en la ruta correcta
Image_path = '27084609001.jpg'

# Llama a la función para obtener las ROIs interactivamente.
# Esta función te mostrará la imagen y te permitirá seleccionarlas.
# Nota: Tendrías que modificar `get_roi_interactively` para que realmente
# retorne la lista de ROIs seleccionadas si vas a usarlas.
# Por ahora, las imprime en consola.
get_roi_interactively(Image_path)

print("\n--- Proceso de OCR ---")

# Si obtuviste las ROIs interactivamente, las usarías aquí.
# Por ejemplo, si get_roi_interactively devuelve una lista de ROIs:
# regions_of_interest = get_roi_interactively(Image_path)
# Si no, las puedes definir manualmente o usar las que se imprimieron en consola:
regions_of_interest = [
    # Ejemplo: Usa las coordenadas que imprimió el selector de ROI
    [100, 50, 400, 200],
    [500, 300, 800, 450]
]


# Resto de tu código (desde la inicialización de EasyOCR hasta el guardado de la imagen)
# ... (Este código ya está en tus versiones anteriores, solo asegúrate de ponerlo aquí) ...

model_storage_directory = '.'
reader = easyocr.Reader(['es'], gpu=False, model_storage_directory=model_storage_directory)

# Realiza la detección de texto en la imagen, usando las regiones de interés.
# Si 'regions_of_interest' está vacío, EasyOCR buscará en toda la imagen por defecto.
result = reader.readtext(Image_path, region=regions_of_interest)

image = cv2.imread(Image_path)
if image is None:
    print(f"Error: No se pudo cargar la imagen desde '{Image_path}'")
    exit()

xyxy, confidences, class_ids, labels = [], [], [], []
excel_data = []

for detection in result:
    bbox, text, confidence = detection[0], detection[1], detection[2]
    x_min = int(min([point[0] for point in bbox]))
    y_min = int(min([point[1] for point in bbox]))
    x_max = int(max([point[0] for point in bbox]))
    y_max = int(max([point[1] for point in bbox]))

    xyxy.append([x_min, y_min, x_max, y_max])
    labels.append(text)
    confidences.append(confidence)
    class_ids.append(0)

    excel_data.append({
        'Texto Detectado': text,
        'Confianza': round(confidence, 4),
        'Coordenada X Mínima': x_min,
        'Coordenada Y Mínima': y_min,
        'Coordenada X Máxima': x_max,
        'Coordenada Y Máxima': y_max
    })

if excel_data:
    df = pd.DataFrame(excel_data)
    excel_output_path = "easyocr_detecciones.xlsx"
    df.to_excel(excel_output_path, index=False)
    print(f"Datos guardados exitosamente en '{excel_output_path}'")
else:
    print("No se detectó texto en la imagen en las regiones especificadas. No se creó ningún archivo de Excel.")

detections = sv.Detections(
    xyxy=np.array(xyxy),
    confidence=np.array(confidences),
    class_id=np.array(class_ids)
)

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image,
    detections=detections,
    labels=labels
)

# Ajuste para mostrar la imagen completa en pantalla
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

cv2.imshow('Imagen Anotada', display_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("Output.jpg", annotated_image)

print("Procesamiento completado. La imagen anotada se ha guardado como Output.jpg")