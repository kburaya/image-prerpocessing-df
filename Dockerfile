FROM python:3.7

ADD models /opt/code/models/
ADD src /opt/code/src/
ADD requirements.txt /opt/code/
RUN pip install -r /opt/code/requirements.txt
RUN python /opt/code/src/server.py prefetch

VOLUME ["/tmp/image-preprocessing/"]
EXPOSE 5000
ENV FLASK_ENV=production
CMD ["python", "/opt/code/src/server.py", "run"]
