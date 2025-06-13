import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk # Necesitarás 'Pillow': pip install Pillow
import easyocr
import supervision as sv
import cv2
import numpy as np
import pandas as pd

# --- Configuración global (ajustable si es necesario) ---
MODEL_STORAGE_DIRECTORY = '.'
READER = None # Se inicializará una vez
REGIONS_OF_INTEREST = [] # Almacenará las ROIs seleccionadas interactivamente
CURRENT_IMAGE_PATH = None
ORIGINAL_IMAGE_FULL_SIZE = None # La imagen original cargada
IMAGE_FOR_DISPLAY = None # La imagen que se muestra en la GUI (puede ser preprocesada o anotada)

# --- Funciones de procesamiento (adaptadas de tu código existente) ---

# Función de callback del ratón para selección de ROI (adaptada para Tkinter o mantener OpenCV)
# NOTA: La selección interactiva de ROI con callback de ratón de OpenCV es difícil
# de integrar directamente en una ventana de Tkinter. Lo más práctico es:
# a) Lanzar una ventana de OpenCV separada solo para la selección de ROI.
# b) Implementar una lógica de dibujo de rectángulos en un lienzo (Canvas) de Tkinter,
#    lo cual es más complejo que el ejemplo básico aquí.
# Para este ejemplo, asumiremos que lanzamos una ventana de OpenCV temporal.
def click_and_crop_opencv(event, x, y, flags, param):
    global REGIONS_OF_INTEREST, ORIGINAL_IMAGE_FULL_SIZE, IMAGE_FOR_DISPLAY, cropping, ref_point_opencv

    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point_opencv = [(x, y)]
        cropping = True
    elif event == cv2.EVENT_MOUSEMOVE and cropping:
        temp_draw_image = IMAGE_FOR_DISPLAY.copy() # Usar la imagen actualmente visible de OpenCV
        cv2.rectangle(temp_draw_image, ref_point_opencv[0], (x, y), (0, 255, 0), 2)
        cv2.imshow("Selecciona ROI - Presiona 'q' para guardar y salir", temp_draw_image)
    elif event == cv2.EVENT_LBUTTONUP:
        ref_point_opencv.append((x, y))
        cropping = False

        x1, y1 = ref_point_opencv[0]
        x2, y2 = ref_point_opencv[1]
        x_min_display, y_min_display = min(x1, x2), min(y1, y2)
        x_max_display, y_max_display = max(x1, x2), max(y1, y2)

        # Opcional: dibujar el ROI final en la imagen mostrada
        cv2.rectangle(IMAGE_FOR_DISPLAY, (x_min_display, y_min_display), (x_max_display, y_max_display), (0, 255, 0), 2)
        cv2.imshow("Selecciona ROI - Presiona 'q' para guardar y salir", IMAGE_FOR_DISPLAY)

        # Convertir coordenadas a la imagen original de tamaño completo
        if ORIGINAL_IMAGE_FULL_SIZE is not None and IMAGE_FOR_DISPLAY is not None:
            original_h, original_w = ORIGINAL_IMAGE_FULL_SIZE.shape[:2]
            display_h, display_w = IMAGE_FOR_DISPLAY.shape[:2]

            scale_x_inv = original_w / display_w
            scale_y_inv = original_h / display_h

            x_min_original = int(x_min_display * scale_x_inv)
            y_min_original = int(y_min_display * scale_y_inv)
            x_max_original = int(x_max_display * scale_x_inv)
            y_max_original = int(y_max_display * scale_y_inv)

            REGIONS_OF_INTEREST.append([x_min_original, y_min_original, x_max_original, y_max_original])
            print(f"ROI agregada (Original): {REGIONS_OF_INTEREST[-1]}")
            root.update_idletasks() # Actualiza la GUI principal si hay un Label para ROIs

# --- Preprocesamiento de imagen (función separada) ---
def preprocess_image(img):
    # Convertir a escala de grises
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Desenfoque Gaussiano
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Binarización adaptativa
    binary_image = cv2.adaptiveThreshold(blurred_image, 255,
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                         25, 10) # Ajusta estos valores

    # Operaciones Morfológicas
    kernel = np.ones((2,2),np.uint8)
    processed_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=1)
    processed_image = cv2.morphologyEx(processed_image, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return processed_image


# --- Funciones de la GUI ---
def select_image():
    global CURRENT_IMAGE_PATH, ORIGINAL_IMAGE_FULL_SIZE, IMAGE_FOR_DISPLAY, REGIONS_OF_INTEREST
    file_path = filedialog.askopenfilename(
        title="Selecciona una imagen",
        filetypes=[("Archivos de imagen", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
    )
    if not file_path:
        return

    CURRENT_IMAGE_PATH = file_path
    REGIONS_OF_INTEREST = [] # Limpia las ROIs anteriores
    update_roi_display() # Actualiza el texto de ROIs en la GUI
    text_output.delete(1.0, tk.END) # Limpia la salida de texto

    ORIGINAL_IMAGE_FULL_SIZE = cv2.imread(CURRENT_IMAGE_PATH)
    if ORIGINAL_IMAGE_FULL_SIZE is None:
        messagebox.showerror("Error", f"No se pudo cargar la imagen desde '{CURRENT_IMAGE_PATH}'")
        return

    # Preparar imagen para mostrar en la GUI (la anotada/procesada en el futuro)
    # Por ahora, muestra la original ajustada para visualización
    display_img_for_gui(ORIGINAL_IMAGE_FULL_SIZE.copy())
    messagebox.showinfo("Imagen Seleccionada", f"'{CURRENT_IMAGE_PATH}' cargada. Ahora selecciona las ROIs.")

def display_img_for_gui(img_cv2):
    global IMAGE_FOR_DISPLAY, photo
    
    # Ajustar para mostrar en el Label de Tkinter
    (h, w) = img_cv2.shape[:2]
    max_display_width = 600 # Ajusta según el tamaño de tu ventana
    max_display_height = 400

    if w > max_display_width or h > max_display_height:
        width_scale = max_display_width / w
        height_scale = max_display_height / h
        scale = min(width_scale, height_scale)

        new_width = int(w * scale)
        new_height = int(h * scale)

        display_image_resized = cv2.resize(img_cv2, (new_width, new_height), interpolation=cv2.INTER_AREA)
    else:
        display_image_resized = img_cv2.copy()

    # Convertir de OpenCV a formato compatible con Pillow (RGB)
    if len(display_image_resized.shape) == 3: # Si es a color
        display_image_rgb = cv2.cvtColor(display_image_resized, cv2.COLOR_BGR2RGB)
    else: # Si es escala de grises o binarizada
        display_image_rgb = cv2.cvtColor(display_image_resized, cv2.COLOR_GRAY2RGB) # Convertir a RGB para PIL

    # Crear imagen para Tkinter
    img_pil = Image.fromarray(display_image_rgb)
    photo = ImageTk.PhotoImage(image=img_pil)
    image_label.config(image=photo)
    image_label.image = photo # Keep a reference!
    IMAGE_FOR_DISPLAY = display_image_resized # Guardar para el callback de OpenCV


def select_roi_interactive():
    global ORIGINAL_IMAGE_FULL_SIZE, IMAGE_FOR_DISPLAY, cropping, ref_point_opencv

    if ORIGINAL_IMAGE_FULL_SIZE is None:
        messagebox.showwarning("Advertencia", "Por favor, selecciona una imagen primero.")
        return

    # Crear una copia de la imagen original para la selección de ROI en OpenCV
    # Redimensionarla para que se ajuste a la pantalla para la selección.
    (h, w) = ORIGINAL_IMAGE_FULL_SIZE.shape[:2]
    max_display_width_opencv = 1200
    max_display_height_opencv = 800

    if w > max_display_width_opencv or h > max_display_height_opencv:
        width_scale = max_display_width_opencv / w
        height_scale = max_display_height_opencv / h
        scale = min(width_scale, height_scale)
        new_width = int(w * scale)
        new_height = int(h * scale)
        IMAGE_FOR_DISPLAY = cv2.resize(ORIGINAL_IMAGE_FULL_SIZE.copy(), (new_width, new_height), interpolation=cv2.INTER_AREA)
    else:
        IMAGE_FOR_DISPLAY = ORIGINAL_IMAGE_FULL_SIZE.copy()


    cv2.namedWindow("Selecciona ROI - Presiona 'q' para guardar y salir")
    cv2.setMouseCallback("Selecciona ROI - Presiona 'q' para guardar y salir", click_and_crop_opencv)

    cropping = False
    ref_point_opencv = []

    messagebox.showinfo("Modo Selección ROI", "Haz clic y arrastra para seleccionar una región. Suelta el clic para confirmarla. Puedes seleccionar múltiples. Presiona 'q' en la ventana de la imagen para cerrar el selector.")

    while True:
        if not cropping: # Solo repintar si no estamos arrastrando para evitar temblores
            cv2.imshow("Selecciona ROI - Presiona 'q' para guardar y salir", IMAGE_FOR_DISPLAY)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyWindow("Selecciona ROI - Presiona 'q' para guardar y salir")
    update_roi_display() # Actualiza el texto en la GUI principal

def update_roi_display():
    if REGIONS_OF_INTEREST:
        roi_text = "ROIs seleccionadas (Original): \n" + "\n".join(str(roi) for roi in REGIONS_OF_INTEREST)
    else:
        roi_text = "Ninguna ROI seleccionada."
    roi_label.config(text=roi_text)


def run_ocr():
    global READER, ORIGINAL_IMAGE_FULL_SIZE, REGIONS_OF_INTEREST

    if CURRENT_IMAGE_PATH is None:
        messagebox.showwarning("Advertencia", "Por favor, selecciona una imagen primero.")
        return
    if not REGIONS_OF_INTEREST:
        messagebox.showwarning("Advertencia", "Por favor, selecciona al menos una ROI primero.")
        return

    # Inicializar EasyOCR Reader una vez
    if READER is None:
        status_label.config(text="Inicializando modelo OCR... Esto puede tardar la primera vez.")
        root.update_idletasks() # Actualizar GUI para mostrar el mensaje
        try:
            READER = easyocr.Reader(['es'], gpu=False, model_storage_directory=MODEL_STORAGE_DIRECTORY)
        except Exception as e:
            messagebox.showerror("Error de EasyOCR", f"No se pudo inicializar EasyOCR: {e}")
            status_label.config(text="Listo")
            return

    status_label.config(text="Preprocesando imagen y ejecutando OCR...")
    root.update_idletasks()

    try:
        # Preprocesar la imagen
        processed_img_for_ocr = preprocess_image(ORIGINAL_IMAGE_FULL_SIZE.copy())

        # Ejecutar OCR en la imagen preprocesada completa
        result = READER.readtext(processed_img_for_ocr, paragraph=False)

        # Filtrar resultados por ROI
        filtered_results = []
        excel_data = []

        def is_point_inside_roi(point, roi):
            x, y = point
            roi_xmin, roi_ymin, roi_xmax, roi_ymax = roi
            return roi_xmin <= x <= roi_xmax and roi_ymin <= y <= roi_ymax

        for detection in result:
            bbox, text, confidence = detection[0], detection[1], detection[2]
            x_min = int(min([p[0] for p in bbox]))
            y_min = int(min([p[1] for p in bbox]))
            x_max = int(max([p[0] for p in bbox]))
            y_max = int(max([p[1] for p in bbox]))

            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2

            is_in_roi = False
            for roi in REGIONS_OF_INTEREST:
                if is_point_inside_roi((center_x, center_y), roi):
                    is_in_roi = True
                    break
            
            if is_in_roi:
                filtered_results.append(detection)
                excel_data.append({
                    'Texto Detectado': text,
                    'Confianza': round(confidence, 4),
                    'Coordenada X Mínima': x_min,
                    'Coordenada Y Mínima': y_min,
                    'Coordenada X Máxima': x_max,
                    'Coordenada Y Máxima': y_max
                })

        text_output.delete(1.0, tk.END) # Limpiar salida anterior
        if filtered_results:
            full_text_output = ""
            xyxy_filtered, confidences_filtered, labels_filtered = [], [], []
            for det in filtered_results:
                full_text_output += f"Texto: {det[1]} (Confianza: {det[2]:.2f})\n"
                xyxy_filtered.append([int(min([p[0] for p in det[0]])), int(min([p[1] for p in det[0]])),
                                      int(max([p[0] for p in det[0]])), int(max([p[1] for p in det[0]]))])
                confidences_filtered.append(det[2])
                labels_filtered.append(det[1])
            text_output.insert(tk.END, full_text_output)
            
            # Anotar y mostrar la imagen
            detections = sv.Detections(
                xyxy=np.array(xyxy_filtered),
                confidence=np.array(confidences_filtered),
                class_id=np.array([0] * len(filtered_results)) # Asumiendo una única clase para texto
            )
            
            box_annotator = sv.BoxAnnotator()
            label_annotator = sv.LabelAnnotator()

            annotated_image = box_annotator.annotate(scene=ORIGINAL_IMAGE_FULL_SIZE.copy(), detections=detections)
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels_filtered)
            
            display_img_for_gui(annotated_image) # Mostrar la imagen anotada
            
            # Guardar Excel (ahora como un botón separado, pero aquí se genera el df)
            df = pd.DataFrame(excel_data)
            global DF_FOR_EXCEL # Almacenar para el botón de guardar
            DF_FOR_EXCEL = df
            save_excel_button.config(state=tk.NORMAL) # Habilitar botón de guardar
            
        else:
            text_output.insert(tk.END, "No se detectó texto en las ROIs seleccionadas.")
            save_excel_button.config(state=tk.DISABLED)

    except Exception as e:
        messagebox.showerror("Error de OCR", f"Ocurrió un error durante el OCR: {e}")
    
    status_label.config(text="Listo")


def save_excel_results():
    global DF_FOR_EXCEL
    if DF_FOR_EXCEL is not None and not DF_FOR_EXCEL.empty:
        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Archivos de Excel", "*.xlsx")],
            title="Guardar resultados de OCR como"
        )
        if file_path:
            try:
                DF_FOR_EXCEL.to_excel(file_path, index=False)
                messagebox.showinfo("Guardado Exitoso", f"Resultados guardados en '{file_path}'")
            except Exception as e:
                messagebox.showerror("Error al Guardar", f"No se pudo guardar el archivo: {e}")
    else:
        messagebox.showwarning("Advertencia", "No hay datos de OCR para guardar.")


# --- Configuración de la Ventana Principal de Tkinter ---
root = tk.Tk()
root.title("Aplicación de OCR Credimara")
root.geometry("1000x800") # Tamaño inicial de la ventana

# Marco para controles de entrada
input_frame = tk.LabelFrame(root, text="Controles de Entrada", padx=10, pady=10)
input_frame.pack(pady=10, padx=10, fill="x")

btn_select_image = tk.Button(input_frame, text="1. Seleccionar Imagen", command=select_image)
btn_select_image.pack(side=tk.LEFT, padx=5, pady=5)

btn_select_roi = tk.Button(input_frame, text="2. Seleccionar ROIs (Ventana Externa)", command=select_roi_interactive)
btn_select_roi.pack(side=tk.LEFT, padx=5, pady=5)

btn_run_ocr = tk.Button(input_frame, text="3. Ejecutar OCR", command=run_ocr)
btn_run_ocr.pack(side=tk.LEFT, padx=5, pady=5)

save_excel_button = tk.Button(input_frame, text="4. Guardar a Excel", command=save_excel_results, state=tk.DISABLED)
save_excel_button.pack(side=tk.LEFT, padx=5, pady=5)

status_label = tk.Label(input_frame, text="Listo", fg="Green")
status_label.pack(side=tk.RIGHT, padx=10)

# Etiqueta para mostrar ROIs seleccionadas
roi_label = tk.Label(root, text="Ninguna ROI seleccionada.", justify=tk.LEFT, wraplength=980)
roi_label.pack(pady=5, padx=10, fill="x")

# Marco para la visualización de la imagen
image_frame = tk.LabelFrame(root, text="Imagen Anotada", padx=5, pady=5)
image_frame.pack(pady=5, padx=10, fill="both", expand=True)

image_label = tk.Label(image_frame)
image_label.pack(expand=True)

# Marco para la salida de texto
output_frame = tk.LabelFrame(root, text="Resultados del OCR", padx=5, pady=5)
output_frame.pack(pady=10, padx=10, fill="both", expand=True)

text_output = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, width=100, height=10)
text_output.pack(fill="both", expand=True)

# Iniciar el bucle de eventos de Tkinter
root.mainloop()