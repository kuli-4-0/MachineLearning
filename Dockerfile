FROM python:3.10.6

COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt

ENV FLASK_APP=main.py

CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]
