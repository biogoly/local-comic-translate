# Local Comic Translate

[English](../README.md) | [한국어](README_ko.md) | [Français](README_fr.md) | [简体中文](README_zh-CN.md) | Español

<img width="1641" height="1001" alt="Interfaz de Comic Translate" src="https://github.com/user-attachments/assets/4cc8c044-b8b4-4773-8c6d-ad7a84cdaca1" />

Local Comic Translate es una bifurcación comunitaria de [Comic Translate](https://github.com/ogkalu2/comic-translate) centrada en la traducción privada y local, y en un flujo de edición de escritorio más seguro y supervisado. Conserva las funciones del proyecto original de detección, OCR, relleno de imágenes (inpainting), renderizado, proyectos, archivos comprimidos y webtoons, y añade traducción con `llama.cpp` gestionado por la aplicación, edición artística con FLUX.2 Klein, APIs directas de proveedores opcionales y mejoras importantes de edición y fiabilidad. No requiere una cuenta, suscripción ni créditos de traducción del servicio del proyecto original.

El sistema de OCR heredado admite inglés, coreano, japonés, francés, chino simplificado y tradicional, ruso, alemán, neerlandés, español e italiano como idiomas de origen; estos y otros idiomas están disponibles como destinos de traducción.

Esta bifurcación está en desarrollo activo. Los cambios descritos a continuación requieren actualmente ejecutar la aplicación desde el código fuente; las descargas del sitio web del proyecto original Comic Translate no los incluyen. Los enlaces de idiomas de arriba ofrecen traducciones de la documentación actual de esta bifurcación. Los nombres de ajustes y botones se mantienen en inglés en estas guías para poder identificarlos entre los distintos idiomas de la interfaz.

## Última actualización — 0.1.0

- **Artistic Edit:** FLUX.2 Klein 4B/9B local con LoRAs compatibles, o la API de Black Forest Labs, con vista previa antes de aplicar los cambios, posibilidad de deshacerlos y conservación de las ediciones en el proyecto.
- **Elección del motor de traducción:** conserva los ajustes de llama-server local mientras cambias a OpenAI, Gemini, Claude o Deepseek con tus propias claves de API.
- **Mejor flujo de trabajo manual:** el OCR funciona directamente sobre regiones dibujadas a mano; las traducciones manuales ya completadas y su formato se conservan al procesar después la página completa; los idiomas de origen y destino se mantienen al cambiar de página.
- **Limpieza localizada:** restaura los píxeles originales con un pincel, clona una muestra fija de color y textura, o rellena una máscara con un color de fondo muestreado.
- **Mejoras visuales del escritorio:** temas Midnight, Parchment, Lavender y Mint, además de Light/Dark, controles de Artistic Edit más claros, una nueva imagen de inicio y catálogos de idiomas de la interfaz actualizados.
- **Identidad independiente de la bifurcación:** ajustes y datos de aplicación separados, sin avisos de actualización ni conexión con el servicio de pago del proyecto original.

## Características destacadas de esta bifurcación

### Traducción privada con LLM, priorizando el procesamiento local

- Un traductor **Local LLM** que no requiere una API de traducción alojada ni una cuenta de Comic Translate.
- El modo **Managed llama.cpp** inicia un proceso local `llama-server`, elige un puerto de bucle local libre, reutiliza el proceso entre solicitudes y lo detiene al cerrar la aplicación.
- El modo **External server** admite endpoints `/v1` compatibles con OpenAI, como los de Ollama, LM Studio o un `llama-server` gestionado por separado.
- Permite configurar el modelo GGUF, el proyector multimodal opcional (`mmproj`), el tamaño del contexto, las capas en GPU, los tiempos de espera de inicio y solicitud, el máximo de tokens de salida, la temperatura, Top P y Top K.
- Entrada opcional de la imagen de la página para modelos con capacidad visual. El OCR existente sigue siendo la fuente principal del texto; la imagen proporciona contexto adicional al modelo multimodal y le permite corregir errores evidentes de OCR.
- La validación estricta de los identificadores de bloque y los reintentos automáticos evitan que una respuesta incompleta o mal formada del modelo asigne silenciosamente las traducciones a bocadillos equivocados.
- La caché de traducción por página utiliza la imagen de origen original, de modo que la limpieza posterior no invalida innecesariamente una traducción almacenada.

### APIs directas de proveedores opcionales

- **Local LLM** sigue siendo la opción predeterminada. Para traducir o realizar OCR no se requiere ni utiliza el inicio de sesión, la suscripción o la compra de créditos del servicio original de Comic Translate.
- En **Settings > Tools > Translator**, elige **OpenAI**, **Google Gemini**, **Anthropic Claude** o **Deepseek** para utilizar directamente un proveedor en la nube.
- En **Settings > Provider APIs**, elige el proveedor correspondiente, introduce tu propia clave de API y selecciona o escribe el identificador del modelo de su API. El proveedor factura directamente a tu cuenta. Los modelos predefinidos son puntos de partida, no una lista completa de los modelos disponibles.
- También se pueden introducir las credenciales de Microsoft Azure OCR y la clave de Gemini para las opciones existentes de OCR en la nube. El OCR predeterminado sigue siendo local.
- **Save Keys** es opcional. Las claves se almacenan en los ajustes locales de esta bifurcación —no en una bóveda cifrada—, nunca en los archivos de proyecto. Desactiva esta opción y guarda los ajustes o cierra la aplicación para eliminar las claves de proveedores almacenadas; las selecciones de modelos se conservan.
- La traducción en la nube envía el texto reconocido y, cuando **Provide Image as Input to AI** está activado y el modelo lo admite, la imagen de la página. Se aplican las políticas y tarifas del proveedor. No se cambia automáticamente de un servicio local a uno en la nube si el primero falla.
- Se ha eliminado la antigua configuración **Advanced / Custom**. Un endpoint Custom guardado y activo se migra a **Local LLM > External server**. Las selecciones anteriores de modelos de nube con nombre se migran a su proveedor directo, conservando el modelo elegido; cada usuario debe aportar su propia clave.
- Las credenciales de los proveedores y la configuración del servidor local son independientes. Cambiar **Tools > Translator** selecciona el motor activo sin borrar ninguna configuración; el desplegable de proveedores de **Provider APIs** solo selecciona qué credenciales editar.

### Edición manual más segura

- Un flujo práctico página por página para revisar las regiones detectadas antes de borrar nada.
- Las selecciones de idioma de origen y destino se mantienen al pasar de una página a otra en modo Manual. Una selección explícita de **Auto** como idioma de origen también se conserva hasta que la cambies.
- La opción **OCR** del menú contextual funciona sobre una región dibujada a mano sin ejecutar primero Detect para toda la página. El OCR local prepara los límites de línea que faltan, y una solicitud explícita de OCR para una región vuelve a reconocerla en vez de devolver una lectura antigua de la caché o procesar otros cuadros.
- Detect sobre la página completa conserva las regiones dibujadas a mano y evita detecciones duplicadas del mismo texto. Las operaciones posteriores de Recognize/Translate respetan el texto de origen y destino ya completado de esas regiones; Segment y Render conservan sus capas de texto y el formato de palabras individuales en vez de eliminarlos o duplicarlos.
- Conservar una capa de texto manual no omite la limpieza del fondo: su región sigue participando en la segmentación y la limpieza. Las regiones manuales aún sin traducir continúan por el flujo normal.
- Controles para acercar, alejar y ajustar la vista, además de atajos de teclado configurables.
- Ordenación natural de páginas por nombre de archivo y ordenación por fecha de modificación.
- El texto de sustitución sin bordes aparece inmediatamente al dibujar una nueva región de texto e introducir texto.
- Borrar y volver a escribir texto conserva la fuente elegida.
- Se pueden aplicar negrita, cursiva y subrayado a palabras seleccionadas, no solo al elemento de texto completo.
- El formato de palabras individuales es compatible con mover y redimensionar los cuadros de texto, y el texto de destino introducido manualmente se conserva al cambiar el foco, incluida la entrada japonesa mediante IME.
- La traducción manual de una página completa crea las capas de texto que falten. Traducir una región seleccionada que aún no se ha renderizado la limpia antes de crear su capa de texto traducido.

### Mejoras de limpieza y de deshacer

- Cada pasada de limpieza manual se puede deshacer y rehacer con los controles habituales Undo/Redo.
- **Restore brush** restaura los píxeles originales de la página base solo bajo el pincel, sin afectar a las ediciones de otras zonas. También restaura las letras originales; no elimina las capas de texto editables superpuestas. Su tamaño se ajusta con el control deslizante de pincel existente. Cada trazo se puede deshacer.
- **Clone brush** (icono de sello): ajusta el tamaño del pincel, haz Alt-clic en una zona limpia y pinta para repetir esa muestra fija de color y textura. El círculo cian discontinuo permanece en el origen; el contorno continuo blanco y negro sigue el destino y se escala con el zoom. **Soft edge** controla la suavidad de la mezcla. **Sample original** está activado de forma predeterminada para tomar la muestra por debajo de los parches de limpieza; desactívalo para copiar la página editada. Al cambiar el tamaño, la página o el modo de origen, debes tomar una nueva muestra con Alt-clic. Cada trazo se puede deshacer; solo cambian los píxeles pintados. No utiliza ningún modelo de IA.
- **Pick color**, seguido de **Fill mask with sampled color**, rellena las zonas pintadas o segmentadas de la página actual con un color de fondo intacto. Utiliza la máscara visible sin ampliación adicional ni inferencia de un modelo. Úsalo para bocadillos de color uniforme; los degradados y el dibujo con textura siguen necesitando inpainting o edición artística. Deshacer restaura tanto los píxeles anteriores como la máscara.
- **Revert All Inpainting on the Current Page** elimina todos los parches de limpieza guardados en una sola acción que se puede deshacer.
- Las limpiezas sucesivas parten de la composición actual de la página, evitando que reaparezca texto eliminado anteriormente en parches posteriores.
- La limpieza manual con pincel admite tanto clics aislados como trazos y utiliza coordenadas enteras seguras para los límites de imagen.
- Las máscaras de respaldo que tienen en cuenta la geometría distinguen los recuadros rectangulares de los bocadillos redondos, reduciendo el texto residual en las esquinas sin dejar de ser conservadoras cerca del dibujo.
- Las ediciones automáticas de página se agrupan en pasos predecibles para deshacer.

### Edición artística con FLUX.2 Klein

- Un proceso de trabajo opcional y aislado de Diffusers permite editar un cuadro de texto seleccionado, una zona pintada o una página completa con FLUX.2 Klein 4B o 9B.
- El editor utiliza la precisión BF16 nativa del modelo, con políticas configurables de GPU o de descarga de componentes a la CPU.
- Se pueden seleccionar LoRAs compatibles de `models/loras/flux2-klein`, con control de intensidad por edición y validación del tamaño de modelo.
- **Use selected translation** permite crear una instrucción **Artistic lettering**, o una instrucción **Empty speech bubble** para colocar después texto editable.
- **Context padding** proporciona al modelo el dibujo circundante como contexto; **Edit margin** amplía la zona que puede modificarse alrededor de un cuadro seleccionado para que una traducción más larga no quede recortada por sus límites originales. Revisa la vista previa antes de aplicar una edición con margen ampliado.
- Los resultados generados aparecen en una vista previa Antes/Después y se aplican como parches de proyecto no destructivos que se pueden deshacer.
- El proceso de FLUX permanece activo, aunque los cambios de modelo o de política de dispositivo provocan su recarga de forma deliberada.
- Las políticas de descarga a CPU permiten seleccionar el índice de GPU física, algo útil cuando otro modelo local ya ocupa una tarjeta.
- Un motor opcional **Black Forest Labs API** permite utilizar Klein 4B y 9B alojados sin un entorno FLUX local ni una GPU. Configura su clave en **Settings > Provider APIs** y selecciona el motor en la parte superior de Artistic Edit. **Local Diffusers** sigue siendo la opción predeterminada en cada inicio.
- Las ediciones en la nube comparten el mismo flujo de selección, contexto, margen de edición, vista previa, Apply y deshacer. Cada generación —incluidas las regeneraciones— solicita permiso para enviar la zona seleccionada más su contexto, o la página completa si se ha elegido esa opción. Una vista previa ya es una generación de pago, aunque la descartes.
- Las LoRAs locales, los pasos, la escala de orientación (guidance) y los controles de GPU no están disponibles para este motor alojado. El procesamiento en la nube se limita a menos de 4 megapíxeles; las salidas se validan antes de devolverlas a las coordenadas de página y limitarlas a la máscara de edición.
- Cancel deja de esperar el resultado en la nube; no garantiza cancelar el trabajo ni los cargos del proveedor. Las solicitudes fallidas o de resultado incierto nunca se vuelven a enviar automáticamente. Tras un tiempo de espera agotado, revisa el panel del proveedor antes de reintentar. Las restricciones de contenido pueden diferir de las de los modelos locales.

Referencia de la API: [edición de imágenes de BFL](https://docs.bfl.ai/flux_2/flux2_image_editing). La integración en la nube cuenta con pruebas automatizadas de transporte y controladores simulados, y durante el desarrollo se ha confirmado el funcionamiento de solicitudes reales a las APIs de Flux y LLM. Esto no garantiza el funcionamiento de todos los proveedores o modelos; el acceso de la cuenta, las políticas de contenido y los cargos dependen del servicio elegido.

Ejemplo de edición artística 1

<img width="3261" height="1980" alt="Ejemplo de edición artística 1" src="https://github.com/user-attachments/assets/32da6bac-01ef-4ed4-a5f0-5fd08772ceec" />

Ejemplo de edición artística 2

<img width="3271" height="2069" alt="Ejemplo de edición artística 2" src="https://github.com/user-attachments/assets/ba29d9fd-8c12-4d2c-8c8f-8854104e088e" />

Ejemplo de edición artística 3

<img width="3273" height="1990" alt="Ejemplo de edición artística 3" src="https://github.com/user-attachments/assets/99f4f497-87b1-405e-a5d4-06dec5b45840" />

### Apariencia del escritorio, idiomas y aislamiento de la bifurcación

- Seis temas: **Dark**, **Light**, **Midnight**, **Parchment**, **Lavender** y **Mint**. Los botones de Artistic Edit tienen fondos adaptados al tema, etiquetas legibles y bordes visibles.
- Nueva imagen de inicio e identidad visual de la bifurcación.
- La interfaz ofrece inglés, coreano, francés, chino simplificado, ruso, japonés, alemán, español e italiano. Las traducciones de las nuevas funciones están incluidas en los nueve catálogos de Qt, incluido el catálogo adicional de turco —que actualmente no se ofrece en el selector de idioma—. Las comprobaciones de cobertura validan los marcadores de posición y los catálogos compilados; los idiomas de la interfaz son independientes de las traducciones del README enlazadas arriba.
- La bifurcación utiliza sus propios ajustes, rutas de datos e identidad de aplicación para evitar interferencias con una instalación separada de la versión original. La migración de datos antiguos solo lee el perfil anterior, sin modificarlo.
- Las comprobaciones automáticas de actualizaciones al iniciar están desactivadas y se ha eliminado la interfaz de venta de cuentas y suscripciones. La opción manual **Check for Updates** consulta ahora las versiones de esta bifurcación en GitHub, no las del proyecto original.

### Mejoras de GPU y fiabilidad

- Las dependencias de Windows incluyen ONNX Runtime GPU con paquetes de ejecución CUDA 13/cuDNN 9 compatibles; TensorRT ya no se prueba salvo que se solicite explícitamente.
- Si CUDA falla, se recurre a la CPU de forma controlada en lugar de exigir TensorRT.
- Las correcciones abarcan la navegación asíncrona entre páginas, la identidad y persistencia de los parches del proyecto, la composición de limpiezas sucesivas y las actualizaciones de renderizado que faltaban.
- Una batería de pruebas de regresión cubre respuestas del LLM local, etiquetas de detección, traducción automática y manual, edición de texto, ordenación de imágenes, geometría y deshacer de la limpieza, caché de traducción y selección de dispositivo.

## Flujo de trabajo recomendado

La traducción automática sigue disponible, pero una restauración cuidadosa de cómics suele beneficiarse de procesar una página cada vez:

1. Carga el cómic y utiliza **Sort** si las páginas no están en orden de lectura.
2. Selecciona el modo **Manual**, establece los idiomas de origen y destino, y pulsa **Detect**. Utiliza **Auto** para el origen cuando la obra realmente mezcle idiomas o desconozcas su idioma.
3. Revisa las regiones antes de procesarlas: elimina las detecciones que pertenezcan al dibujo y dibuja cuadros alrededor de los diálogos o textos de apoyo que falten.
4. Pulsa **Recognize** y corrige el texto de origen si el OCR se ha equivocado.
5. Pulsa **Translate** y revisa o edita el texto de destino.
6. Pulsa **Segment**, inspecciona las zonas de limpieza propuestas y después pulsa **Clean**.
7. Pulsa **Render** y ajusta la fuente, el tamaño, la alineación, el espaciado y el énfasis de cada elemento de texto.
8. Utiliza el pincel de limpieza y **Apply Cleanup** para eliminar las marcas restantes. Si la limpieza daña el dibujo, puedes usar Undo/Redo o **Revert All Inpainting on the Current Page**.

También puedes dibujar una región que se haya omitido y elegir **OCR** y después **Translate** con el botón derecho, sin detectar primero el resto de la página. Su traducción completada se conserva si después ejecutas el flujo manual de página completa. En fondos de color o con textura, prueba el relleno con color muestreado, el pincel de restauración o el de clonado antes de repetir el inpainting sobre la misma zona. Utiliza Artistic Edit para la rotulación que deba formar parte del dibujo en lugar de constituir una capa de texto editable normal.

El detector conserva intencionadamente tanto el texto de bocadillos como el texto situado fuera de ellos. Esto evita perder regiones válidas de OCR en el modo manual, pero también significa que **Translate All** puede borrar mediante inpainting títulos artísticos, carteles, onomatopeyas u otras letras que formen parte del dibujo. Se recomienda revisar manualmente las regiones antes de limpiarlas cuando sea importante preservar la ilustración.

## Instalación

### Requisitos

- Python 3.12
- [Git](https://git-scm.com/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Un ejecutable local `llama-server` y un modelo GGUF compatible si deseas utilizar la traducción local gestionada por la aplicación
- WinRAR o 7-Zip en `PATH` para abrir archivos CBR

### Ejecutar desde el código fuente

```bash
git clone https://github.com/biogoly/local-comic-translate.git
cd local-comic-translate
uv init --python 3.12
uv add -r requirements.txt --compile-bytecode
uv run comic.py
```

En Windows, también se proporciona `run.bat` después de la instalación inicial de dependencias:

```bat
run.bat
```

### Entorno de ejecución opcional de FLUX.2 Klein

El motor **Local Diffusers** utiliza un entorno Python separado para que sus versiones más recientes de PyTorch, Diffusers y Transformers no afecten al entorno principal de la aplicación. El motor BFL API no requiere este entorno opcional. Desde la raíz del repositorio, crea un entorno Python 3.12 e instala las versiones fijadas de las dependencias opcionales:

```bat
py -3.12 -m venv .venv-flux
.venv-flux\Scripts\python.exe -m pip install -r requirements-flux.txt
```

En **Artistic Edit**, pulsa **Select FLUX Python…** y selecciona `.venv-flux\Scripts\python.exe`. Los archivos de modelo ya deben encontrarse en la caché de Hugging Face, porque el proceso integrado trabaja de forma predeterminada solo con archivos en caché. El aislamiento del perfil de la bifurcación no crea otra caché de Hugging Face ni duplica las rutas de modelos seleccionadas por el usuario.

Coloca las LoRAs `.safetensors` opcionales en `models/loras/flux2-klein/4b/` o `models/loras/flux2-klein/9b/`, según corresponda. Consulta la [guía de la carpeta LoRA](../models/loras/flux2-klein/README.md) para conocer la compatibilidad y los metadatos opcionales. Los pesos no se incluyen en el repositorio. El proceso local actual utiliza BF16; la carga experimental INT8/NF4 con bitsandbytes no está activada.

Para actualizar una copia existente del repositorio:

```bash
git pull
uv add -r requirements.txt --compile-bytecode
```

### Configurar un LLM local

`llama.cpp` es un entorno de ejecución externo y no se instala mediante `requirements.txt` ni `uv`. Descarga o compila [`llama.cpp`](https://github.com/ggml-org/llama.cpp) y obtén un modelo GGUF de conversación o seguimiento de instrucciones que quepa en tu hardware.

El motor no está vinculado a un modelo concreto; Gemma 4 es el modelo principal utilizado durante el desarrollo y las pruebas actuales de esta bifurcación.

En la aplicación:

1. Abre **Settings > Tools** y selecciona **Local LLM** como traductor.
2. Abre **Settings > LLMs**.
3. Elige uno de estos modos de ejecución:
   - **Managed llama.cpp**: selecciona `llama-server` —opcional si ya está en `PATH`—, el modelo GGUF principal y el proyector visual GGUF correspondiente si el modelo lo requiere.
   - **External server**: introduce la URL base compatible con OpenAI y el nombre exacto del modelo que expone el servidor. La URL predeterminada, `http://127.0.0.1:11434/v1`, corresponde a Ollama.
4. Activa **Provide Image as Input to AI** solo si el modelo y el servidor elegidos admiten imágenes. En modo gestionado, configura primero el archivo `mmproj` requerido.

Los valores conservadores predeterminados de traducción son Temperature `0.20`, Top P `0.90` y Top K `40`. Son buenos puntos de partida para traducir; ajústalos solo si la documentación de tu modelo recomienda otros valores de muestreo. Establece Top K en `0` para desactivarlo.

### Alternar entre motores locales y APIs

Para la traducción de texto:

1. Abre **Settings > Provider APIs**, elige el proveedor, introduce su clave y selecciona o escribe el identificador del modelo. Activa **Save Keys** solo si quieres conservar la clave entre sesiones.
2. Abre **Settings > Tools > Translator** y elige ese proveedor. El cambio se aplica a las siguientes solicitudes sin reiniciar.
3. Vuelve a seleccionar **Local LLM** para regresar a la configuración guardada de llama.cpp o del servidor externo. No es necesario sustituir el endpoint local ni borrar sus ajustes de modelo.

Para la edición artística, introduce la clave de **Black Forest Labs** en **Provider APIs** y cambia el desplegable superior de **Artistic Edit** de **Local Diffusers** a **Black Forest Labs API**. Elige 4B/9B y pulsa **Generate Preview**. Al volver al modo local recuperas el acceso a sus controles. Esta selección es independiente del traductor de texto.

Al probar otro proveedor, utiliza una página o región sin traducir: las traducciones manuales ya completadas se conservan intencionadamente y los resultados en caché pueden evitar una solicitud nueva. Las solicitudes en la nube envían datos al proveedor elegido y pueden generar cargos; una vista previa artística descartada sigue siendo una generación.

### Aceleración por GPU

Existen dos ajustes de GPU independientes:

- **Settings > Tools > Use GPU** controla las operaciones compatibles de ONNX/PyTorch, como detección, OCR e inpainting. En Windows, las dependencias instalan ONNX Runtime GPU y los paquetes de ejecución CUDA 13/cuDNN 9 correspondientes. Sigue siendo necesario un controlador NVIDIA compatible, pero no una instalación separada de CUDA Toolkit o TensorRT.
- **Settings > LLMs > GPU layers** controla cuántas capas intenta transferir a la GPU el `llama.cpp` gestionado. Los servidores externos gestionan sus propios ajustes de GPU.

El procesamiento por CPU sigue disponible como alternativa. La mejora de velocidad con GPU depende del modelo, el tamaño de página y la etapa que limite el rendimiento; la velocidad del flujo de página completa no tiene por qué aumentar proporcionalmente al uso bruto de GPU.

## Consejos de uso

- `Ctrl` + rueda del ratón: zoom
- `Ctrl` + `=`: acercar
- `Ctrl` + `-`: alejar
- `Ctrl` + `0`: ajustar la página a la ventana
- Flecha izquierda/derecha: cambiar de página
- El visor de imágenes admite los gestos habituales del panel táctil
- Asegúrate de que la fuente elegida admita el idioma de destino
- Las opciones de ordenación son **Name: A to Z**, **Name: Z to A**, **Modified: Oldest First** y **Modified: Newest First**. La ordenación por nombre es natural, por lo que `page2` aparece antes que `page10`.
- Si un archivo CBR provoca `RarCannotExec("Cannot find working tool")`, añade la carpeta de instalación de WinRAR o 7-Zip a `PATH`.

## Cómo funciona

### Detección de bocadillos y segmentación de texto

La aplicación utiliza el [detector de bocadillos y texto](https://huggingface.co/ogkalu/comic-text-and-bubble-detector) del proyecto original, un modelo RT-DETR-v2 entrenado con manga, webtoons y cómics occidentales. Los resultados de detección se pueden editar antes del OCR, la segmentación o la limpieza.

<img src="https://i.imgur.com/TlzVH3j.jpg" width="49%" alt="Texto de cómic detectado"> <img src="https://i.imgur.com/h18XrYT.jpg" width="49%" alt="Texto de cómic segmentado">

### OCR

Los motores OCR predeterminados son:

- [manga-ocr](https://github.com/kha-white/manga-ocr) para japonés
- [Pororo](https://github.com/yunwoong7/korean_ocr_using_pororo) para coreano
- [PPOCRv5](https://www.paddleocr.ai/main/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5.html) para los demás idiomas compatibles

Gemini y Microsoft Azure Vision siguen disponibles como integraciones opcionales de OCR directo con el proveedor, utilizando tus propias credenciales.

### Traducción

La traducción utiliza la conexión local compatible con OpenAI o las integraciones directas de proveedores con claves del usuario descritas arriba; no pasa por el servicio de pago del proyecto original. Las solicitudes de página completa agrupan las regiones reconocidas que deben procesarse para aportar contexto. Se conservan las regiones manuales completadas, y las solicitudes de una región seleccionada pueden usar resultados en caché o enviar una petición más limitada. Los modelos multimodales compatibles también pueden recibir opcionalmente la imagen de la página.

### Limpieza y edición artística

Los motores locales de limpieza siguen siendo LaMa, MI-GAN y AOT. La herramienta separada Artistic Edit de FLUX.2 Klein realiza cambios generativos, como traducir rotulación artística, reconstruir o redimensionar un bocadillo y reparar zonas concretas del dibujo. Es un flujo supervisado de vista previa y aplicación, no un sustituto automático de la limpieza.

<img src="https://i.imgur.com/cVVGVXp.jpg" width="49%" alt="Cómic antes de la limpieza"> <img src="https://i.imgur.com/bLkPyqG.jpg" width="49%" alt="Cómic después de la limpieza">

### Renderizado de texto

El texto traducido o introducido manualmente se distribuye dentro de regiones editables. Antes de exportar, se pueden ajustar la fuente, el tamaño, la alineación, el espaciado, el color, el contorno, la dirección y el énfasis a nivel de caracteres.

## Pruebas

Ejecuta las pruebas de regresión desde la raíz del repositorio:

```bash
uv run python -m unittest discover -s tests -v
```

Comprueba por separado la cobertura de traducciones de la interfaz y los catálogos compilados:

```bash
uv run python resources/translations/check_translations.py
```

Las comprobaciones de publicación incluyen OCR y conservación de regiones manuales, selección de proveedores locales/API, contratos y vistas previas de edición artística, persistencia de parches de proyecto, deshacer de limpieza y clonado, temas, persistencia de idiomas y aislamiento del perfil de la bifurcación. En las pruebas de regresión se simula la mayoría de las operaciones de modelos y red; no descargan pesos ni consumen créditos de API.

## Limitaciones actuales

- Esta bifurcación está en una fase inicial y solo se distribuye como código fuente; continúan las pruebas amplias con distintos idiomas, modelos, diseños de página y sistemas operativos.
- El modo automático no siempre distingue los diálogos de la rotulación que forma parte del dibujo. Utiliza el modo Manual cuando sea importante preservar la ilustración.
- La detección, el OCR, la traducción, la segmentación y el inpainting pueden requerir corrección humana en páginas difíciles.
- La entrada visual requiere un modelo realmente multimodal, un servidor compatible y, cuando corresponda, el archivo de proyector adecuado.
- La calidad y la velocidad de la generación local dependen en gran medida del modelo, del tamaño de contexto y de la RAM/VRAM disponible.
- Artistic Edit requiere actualmente el modo de página única; no es una etapa automática para libros completos ni un editor de varias páginas de webtoon. Antes de aplicar la rotulación generada, revisa su ortografía y comprueba que no haya cambios no deseados en el dibujo.

## Ejemplos de cómics del proyecto original

Estos ejemplos muestran el flujo original de Comic Translate con GPT alojado. Algunos títulos también tienen traducciones oficiales al inglés.

- [The Wretched of the High Seas](https://www.drakoo.fr/bd/drakoo/les_damnes_du_grand_large/les_damnes_du_grand_large_-_histoire_complete/9782382330128)
- [Journey to the West](https://ac.qq.com/Comic/comicInfo/id/541812)
- [The Wormworld Saga](https://wormworldsaga.com/index.php)
- [Frieren: Beyond Journey's End](https://renta.papy.co.jp/renta/sc/frm/item/220775/title/742932/)
- [Days of Sand](https://9ekunst.nl/2021/05/20/nieuw-album-van-aimee-de-jongh-is-benauwd-als-een-zandstorm/)
- [Player (OH Hyeon-Jun)](https://comic.naver.com/webtoon/list?titleId=745876&page=1&sort=ASC&tab=fri)
- [Carbon & Silicon](https://www.amazon.com/Carbone-Silicium-French-Mathieu-Bablet-ebook/dp/B0C1LTGZ85/)

## Agradecimientos

Este trabajo es una bifurcación de [ogkalu2/comic-translate](https://github.com/ogkalu2/comic-translate). La aplicación original, sus modelos, interfaz e integraciones son la base de los cambios documentados aquí.

- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [lama-cleaner](https://github.com/Sanster/lama-cleaner)
- [dreMaz/AnimeMangaInpainting](https://huggingface.co/dreMaz/AnimeMangaInpainting)
- [Pororo Korean OCR](https://github.com/yunwoong7/korean_ocr_using_pororo)
- [manga-ocr](https://github.com/kha-white/manga-ocr)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [RapidOCR](https://github.com/RapidAI/RapidOCR)
- [dayu_widgets](https://github.com/phenom-films/dayu_widgets)

Publicado bajo la [licencia Apache 2.0](../LICENSE).
