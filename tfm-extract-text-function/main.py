import functions_framework
import json
import os
import tempfile
from typing import Optional
from google.api_core.client_options import ClientOptions
from google.cloud import documentai, storage  # type: ignore
import google.auth
import logging

credentials, project = google.auth.default()
logger = logging.getLogger()

PROJECT_ID = os.environ.get('PROJECT_ID')
LOCATION = os.environ.get('LOCATION') # Format is 'us' or 'eu'
PROCESSOR_ID = os.environ.get('PROCESSOR_ID')
PDF_MIME_TYPE = os.environ.get('PDF_MIME_TYPE')
FIELD_MASK = os.environ.get('FIELD_MASK')  # Optional. The fields to return in the Document object.
EXPECTED_SUFFIX_EXTENSION = os.environ.get('EXPECTED_SUFFIX_EXTENSION')
PDF_FILES_BUCKET_NAME = os.environ.get('PDF_FILES_BUCKET_NAME')
TXT_RAW_FILES_BUCKET_NAME = os.environ.get('TXT_RAW_FILES_BUCKET_NAME')

# Triggered by a change in a storage bucket
@functions_framework.cloud_event
def handler(cloud_event):
    data = cloud_event.data
    bucket = data["bucket"]
    file_key = data["name"]

    #file_key = (event['attributes']['file_key'])
    logger.info(f'El fileKey recibido en el mensaje es: {file_key}')
    # Verificamos primero si el fichero que ha llegado tiene la extensión que estamos monitorizando
    if file_key != None and len(file_key) > len(EXPECTED_SUFFIX_EXTENSION) and file_key.find(EXPECTED_SUFFIX_EXTENSION, len(file_key) - len(EXPECTED_SUFFIX_EXTENSION)) != -1:
        error = do_something(bucket, file_key)
    else:
        logger.info('Nothing to do...')

    if not error:
        return {
            'statusCode': 200,
            'body': json.dumps('Function finished successfully!')
        }
    else:
        raise('Errors found during execution :(')

def do_something(bucket, file_key):
    logger.info(f'Processing input matadata file {file_key} in bucket {bucket}')

    # Descargamos primero el fichero y leemos su contenido
    _, temp_local_filename = tempfile.mkstemp()
    metadata = download_blob(
        bucket_name = bucket,
        source_blob_name = file_key,
        destination_file_name = temp_local_filename)

    logger.info('Datos leidos del fichero {}:'.format(file_key))
    logger.info(json.dumps(metadata))

    nombre_fichero_pdf = f"{metadata['cod_documento']}.pdf"
    nombre_fichero_txt = f"{metadata['cod_documento']}.json"

    # Verificamos si se trata de una ejecución manual desde texto para omitir el paso de extracción de texto
    error = False
    if not 'unique_req_text' in metadata:
        logger.info(f'Se procede a extraer el texto del fichero {nombre_fichero_pdf}, que será escrito en {nombre_fichero_txt}')
        try:
            async_detect_document(nombre_fichero_pdf, nombre_fichero_txt, metadata)
        except Exception as e:
            # Si hay cualquier problema, se encola en el topic de errores
            logger.error(f'Error al invocar Document AI API: {e}')
            error = True
    else:
        logger.info('Se ignora el fichero de metadatos, pues representa una ejecución manual a partir de texto')
    
    return error


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    with open(destination_file_name, 'wb') as file_obj:
        blob.download_to_file(file_obj)
    
    with open(destination_file_name, 'rb') as file_obj:
        metadata = json.load(file_obj)

    logger.info('Blob {} downloaded successfully'.format(source_blob_name))
    return metadata


def async_detect_document(pdf_filename, target_json_filename, metadata):
    """OCR with PDF/TIFF as source files on GCS"""
    source_bucket_name = PDF_FILES_BUCKET_NAME
    file_path = f"gs://{source_bucket_name}/{pdf_filename}"
    logger.info(f"Path del fichero PDF a procesar: {file_path}")

    storage_client = storage.Client()
    source_bucket = storage_client.bucket(source_bucket_name)
    blob = source_bucket.blob(pdf_filename)

    document = process_document_sample(
        project_id=PROJECT_ID,
        location=LOCATION,
        processor_id=PROCESSOR_ID,
        blob=blob,
        mime_type=PDF_MIME_TYPE,
        field_mask=FIELD_MASK
    )

    extraccion = {}
    # Añadimos los metadatos que ya teníamos del JSON que disparó el evento
    extraccion['id_legislatura'] = metadata['id_legislatura']
    extraccion['camara'] = 1 if metadata['id_legislatura'] == 184 else 2
    extraccion['num_registro'] = metadata['num_registro']
    extraccion['cod_documento'] = metadata['cod_documento']
    extraccion['texto'] = document.text

    # Extracción RAW
    write_to_gcs(extraccion, storage_client, TXT_RAW_FILES_BUCKET_NAME, target_json_filename)
    logger.info(f'PDF File gs://{source_bucket_name}/{pdf_filename} successfully sent to Document AI API and RAW results written to gs://{TXT_RAW_FILES_BUCKET_NAME}/{target_json_filename}')


def write_to_gcs(extraccion, storage_client, target_bucket_name, target_json_filename):
    target_bucket = storage_client.bucket(target_bucket_name)
    blob = target_bucket.blob(target_json_filename)
    with blob.open("w") as f:
        f.write(json.dumps(extraccion))


def process_document_sample(
    project_id: str,
    location: str,
    processor_id: str,
    blob,
    mime_type: str,
    field_mask: Optional[str] = None,
) -> documentai.Document:
    # You must set the api_endpoint if you use a location other than 'us'.
    opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")

    client = documentai.DocumentProcessorServiceClient(client_options=opts)

    # The full resource name of the processor, e.g.:
    # projects/{project_id}/locations/{location}/processors/{processor_id}
    name = client.processor_path(project_id, location, processor_id)

    # Read the file into memory
    with blob.open("rb") as image:
        image_content = image.read()

    # Load Binary Data into Document AI RawDocument Object
    raw_document = documentai.RawDocument(content=image_content, mime_type=mime_type)

    # Configure the process request
    request = documentai.ProcessRequest(
        name=name, raw_document=raw_document, field_mask=field_mask
    )

    result = client.process_document(request=request)
    document = result.document

    # Read the text recognition output from the processor
    logger.info("The document contains the following text:")
    logger.info(document.text)
    return document
