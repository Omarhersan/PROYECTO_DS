import uvicorn
from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from fastapi.exceptions import HTTPException
import os
import shutil
from PIL import Image
from pydantic import BaseModel

# Loading model and initializing the fastapi app
model = YOLO('best.pt')
app = FastAPI()


@app.post('/api/v0/upload_file')
async def upload_file(file: UploadFile):
    content_type = file.content_type
    if content_type not in ["image/jpeg", "image/png", "image/gif"]:
        raise HTTPException(status_code=400, detail="Invalid file type")

    upload_dir = os.path.join(os.getcwd(), "uploads")
    # Create the upload directory if it doesn't exist
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    # get the destination path
    dest = os.path.join(upload_dir, file.filename)
    # copy the file contents
    with open(dest, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"filename": file.filename}



@app.get('/api/v0/get_prediction')
async def get_prediction(filename: str):
    file_path = os.path.join(os.getcwd(), f'uploads/{filename}')
    results = model.predict(file_path)
    # Show the results
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        im.show()  # show image
        if not os.path.exists(os.path.join(os.getcwd(), 'results')):
            os.makedirs(os.path.join(os.getcwd(), 'results'))

        im.save(f'results/{filename}')  # save image


    return {"Done": "Done"}












if __name__ == '__main__':
    uvicorn.run('main:app',host="0.0.0.0", port = 5000, log_level = "info", reload = False)





