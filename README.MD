# CBIR
El objetivo principal de este proyecto es implementar un
sistema CBIR aplicado a paisajes, empleando un conjunto
de datos que incluye cinco clases visuales: glaciares, costa,
desierto, montaña y bosque. Para ello, se utilizan diferentes
métodos de extracción de características, desde técnicas
tradicionales como histogramas de color, GLCM y HOG,
hasta técnicas avanzadas basadas en aprendizaje profundo,
como embeddings de CNN y transformers. 

El desempeño de cada método ha sido evaluado, permitiendo comparar su
eficacia en términos de precisión y recuperación. A través
de este análisis, se busca identificar el enfoque más efectivo
para la tarea de recuperación de imágenes, contribuyendo al
desarrollo de sistemas CBIR más eficientes y prácticos.

### Notas importantes
- Para ejecutar la aplicación correctamente, es necesario contar con el modelo CNN, que debe estar en el directorio principal. Se puede obtener el modelo descargándolo desde el siguiente enlace de Google Drive: https://drive.google.com/file/d/1h_h_FCyDz-XQuEYm-gRbbEDVLGHJerMN/view?usp=sharing, o bien ejecutar el notebook de creación llamado CNN.ipynb, donde también se puede modificar la red al gusto de cada uno.
- El modelo de transformers se carga directamente desde Hugging Face. Esto implica que, al lanzar la aplicación, tarda unos 10-15 segundos en cargar la página mientras el modelo se está cargando. 

## Requisitos

- Python >= 3.8

## Instalación
1. Clona este repositorio en tu máquina local:

    ```bash
    git clone https://github.com/upm-classes/aapi_2024
    cd cbir
    ```
2. Crea un entorno virtual (opcional pero recomendado):

    ```bash
    python -m venv cbir
    ```
    
    En Windows :
    ```
    cbir\Scripts\activate
    ```
    
    En macOS/Linux :
    ```
    source cbir/bin/activate  
    ```

3. Instala las dependencias:

    ```bash
    pip install -r requirements.txt
    ```

## Ejecución

Para ejecutar la aplicación, usa el siguiente comando:

```bash
streamlit run app.py
```
La aplicación se ejecutará en http://localhost:8501

### Nota importante
Para ejecutar la aplicación correctamente, es necesario contar con el modelo CNN, que debe estar en el directorio principal. Se puede obtener el modelo descargándolo desde el siguiente enlace de Google Drive: https://drive.google.com/file/d/1h_h_FCyDz-XQuEYm-gRbbEDVLGHJerMN/view?usp=sharing, o bien ejecutar el notebook de creación llamado CNN.ipynb, donde también se puede modificar la red al gusto de cada uno.
