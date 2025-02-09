FROM python:3.11
WORKDIR /app
COPY . /app
RUN python -m pip install --upgrade pip
RUN apt-get update && apt-get install -y libopenblas-dev
RUN pip install --no-cache-dir -v -r requirements.txt
# CMD ["python"]
CMD ["streamlit", "run", "app.py"]