import os
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {"csv"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def save_file(file, upload_folder):
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(upload_folder, filename)
        file.save(filepath)
        return filepath
    else:
        raise ValueError("Arquivo inválido. Envie um CSV.")
