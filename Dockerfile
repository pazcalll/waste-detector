
FROM python:3.10


WORKDIR /code


COPY . /code

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN pip install --upgrade opencv-python
RUN pip install --upgrade opencv-contrib-python
RUN pip install fastapi[standard] uvicorn

#COPY ./app /code/app

ENV PORT=8001
EXPOSE 8001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]