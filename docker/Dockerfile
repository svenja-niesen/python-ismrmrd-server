FROM ubuntu:20.04 AS mrd_converter
LABEL maintainer="Philipp Ehses (philipp.ehses@dzne.de)"

ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update --quiet \
    && apt-get install --no-install-recommends --no-install-suggests --yes \
       apt-transport-https ca-certificates libhdf5-serial-dev h5utils cmake cmake-curses-gui libboost-all-dev libfftw3-dev git build-essential gfortran libpng-dev liblapacke-dev libpng-dev libopenblas-dev libxml2-dev libxslt1-dev 

RUN update-ca-certificates

RUN  mkdir -p /opt/code

# ISMRMRD library
RUN cd /opt/code \
    && git clone https://github.com/ismrmrd/ismrmrd.git \
    && cd ismrmrd \
    && mkdir build \
    && cd build \
    && cmake ../ \
    && make -j $(nproc) \
    && make install

# siemens_to_ismrmrd converter
RUN cd /opt/code \
    && git clone https://github.com/ismrmrd/siemens_to_ismrmrd.git \
    && cd siemens_to_ismrmrd \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make -j $(nproc) \
    &&  make install


# compile & install the bart MRI toolbox
RUN  cd /opt/code \
     && git clone https://github.com/mrirecon/bart \
     && cd bart \
     && make -j $(nproc) \
     && make install
     # && echo ISMRMRD=1 > Makefile.local \
     # && export ISMRMRD_HOME=/usr/local \

# ----- Start another clean build without all of the build dependencies of siemens_to_ismrmrd -----
FROM ubuntu:20.04
LABEL maintainer="Philipp Ehses (philipp.ehses@dzne.de)"

ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Copy siemens_to_ismrmrd from last stage and re-add necessary dependencies
COPY --from=mrd_converter /usr/local/bin/siemens_to_ismrmrd /usr/local/bin/siemens_to_ismrmrd
COPY --from=mrd_converter /usr/local/bin/bart /usr/local/bin/bart
COPY --from=mrd_converter /usr/local/lib/libismrmrd.so.1.4.* /usr/local/lib/
RUN  ln -s libismrmrd.so.1.4.* libismrmrd.so.1.4
RUN  ln -s libismrmrd.so.1.4 libismrmrd.so

# Dependencies for Python MRD server
RUN  apt-get update --quiet \
     && apt-get dist-upgrade --yes \
     && apt-get install --no-install-recommends --no-install-suggests --yes \
        apt-transport-https ca-certificates python3 python3-pip h5utils libxslt1.1 git gcc libfftw3-3 liblapacke\
     && pip3 install --no-cache-dir pyxb h5py

RUN  update-ca-certificates

RUN  mkdir -p /opt/code

RUN  cd /opt/code \
     && git clone https://github.com/ismrmrd/ismrmrd-python.git \
     && cd /opt/code/ismrmrd-python \
     && python3 setup.py install

RUN  cd /opt/code \
     && git clone https://github.com/ismrmrd/ismrmrd-python-tools.git \
     && cd /opt/code/ismrmrd-python-tools \
     && python3 setup.py install

RUN  cd /opt/code \
     && git clone -b bart https://github.com/pehses/python-ismrmrd-server.git

RUN  cd /opt/code \
     && git clone https://github.com/mrirecon/bart

ENV PYTHONPATH "${PYTHONPATH}:/opt/code/bart/python"

# Cleanup files not required after compiling
RUN  apt-get remove --yes gcc git
RUN  apt-get autoremove --yes 
RUN  apt-get autoclean --yes
RUN  rm -r /root/.cache/pip

# CMD [ "/usr/local/bin/python3", "/opt/code/python-ismrmrd-server/main.py", "-v", "-H=0.0.0.0", "-p=9002", "-l=/tmp/python-ismrmrd-server.log"]
CMD [ "python3", "/opt/code/python-ismrmrd-server/main.py", "-v", "-H=0.0.0.0", "-p=9002", "-l=/tmp/python-ismrmrd-server.log"]
