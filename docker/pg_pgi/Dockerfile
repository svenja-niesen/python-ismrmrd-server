FROM ubuntu:20.04 AS mrd_converter

LABEL maintainer="Philipp Ehses (philipp.ehses@dzne.de)"

ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update --quiet \
    && apt-get install --no-install-recommends --no-install-suggests --yes \
       wget apt-utils apt-transport-https gnupg2 python3-dev python3-pip ca-certificates libhdf5-serial-dev h5utils cmake cmake-curses-gui libboost-all-dev libfftw3-dev git build-essential gfortran libpng-dev liblapacke-dev libpng-dev libopenblas-dev libxml2-dev libxslt1-dev libssl-dev curl

# install mkl
COPY ./mkl.sh /tmp/
RUN /tmp/mkl.sh


RUN  mkdir -p /opt/code

RUN  cd /opt/code \
     && git clone -b bart https://github.com/pehses/python-ismrmrd-server.git

# install spiraltraj for trajectory calculation
COPY spiraltraj /opt/code/spiraltraj
RUN  cd /opt/code \
     && cd spiraltraj \
     && pip3 wheel .
     # && git clone https://github.com/mrphysics-bonn/spiral-pypulseq-example \
     # && mv spiral-pypulseq-example/spiraltraj . \


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
    && cp /opt/code/python-ismrmrd-server/parameter_maps/* parameter_maps/ \
    && mkdir build \
    && cd build \
    && cmake ../ \
    && make -j $(nproc) \
    && make install


# compile & install the bart MRI toolbox
RUN  cd /opt/code \
     && git clone https://github.com/mrirecon/bart \
     && cd bart \
     && echo MKL=1 >> Makefile.local \
     && echo MKL_BASE=/opt/intel/mkl >> Makefile.local \
     && echo PARALLEL=1 >> Makefile.local \
     && make \
     && make install
     # && echo SLINK=1 >> Makefile.local \
     # && echo CUDA=1 >> Makefile.local \
     # && echo CUDA_BASE=/usr/local/cuda/ >> Makefile.local \
     # && echo ENABLE_MEM_CFL=1 >> Makefile.local \
     # && echo ISMRMRD=1 > Makefile.local \
     # && export ISMRMRD_HOME=/usr/local \

### Dependencies for PowerGrid ###

# Build and Install Latest CMake
RUN curl -O -J -L https://github.com/Kitware/CMake/releases/download/v3.19.2/cmake-3.19.2.tar.gz \
    && apt-get -y remove --purge cmake \
    && tar -xvf ./cmake-3.19.2.tar.gz \
    && cd ./cmake-3.19.2 \
    && ./bootstrap --prefix=/usr/local/ \
    && make -j$(nproc) \
    && make install

# Download and install NVIDIA HPC SDK
RUN  wget https://developer.download.nvidia.com/hpc-sdk/20.11/nvhpc-20-11_20.11_amd64.deb \
     https://developer.download.nvidia.com/hpc-sdk/20.11/nvhpc-2020_20.11_amd64.deb \
     && apt-get install ./nvhpc-20-11_20.11_amd64.deb ./nvhpc-2020_20.11_amd64.deb

# Handle NVIDIA HPC SDK specific stuff
ENV NVARCH=Linux_x86_64
ENV NVCOMPILERS=/opt/nvidia/hpc_sdk
ENV NVSDKVER=20.11
ENV CUDA_PATH="${NVCOMPILERS}/${NVARCH}/${NVSDKVER}/cuda"
ENV MANPATH="${MANPATH}:${NVCOMPILERS}/${NVARCH}/20.11/compilers/man"
ENV LD_LIBRARY_PATH="${CUDA_PATH}/lib64:${LD_LIBRARY_PATH}"
ENV LD_LIBRARY_PATH="${NVCOMPILERS}/${NVARCH}/20.11/compilers/lib:${LD_LIBRARY_PATH}"
ENV LD_LIBRARY_PATH="${NVCOMPILERS}/${NVARCH}/2020/compilers/lib:${LD_LIBRARY_PATH}"
ENV LD_LIBRARY_PATH="${NVCOMPILERS}/${NVARCH}/20.11/math_libs/lib64:${LD_LIBRARY_PATH}"
ENV LD_LIBRARY_PATH="${NVCOMPILERS}/${NVARCH}/2020/math_libs/lib64:${LD_LIBRARY_PATH}"
ENV LD_LIBRARY_PATH="${NVCOMPILERS}/${NVARCH}/20.11/comm_libs/mpi/lib:${LD_LIBRARY_PATH}"
ENV LD_LIBRARY_PATH="${NVCOMPILERS}/${NVARCH}/2020/comm_libs/mpi/lib:${LD_LIBRARY_PATH}"
ENV LD_LIBRARY_PATH="${NVCOMPILERS}/${NVARCH}/20.11/comm_libs/nccl/lib:${LD_LIBRARY_PATH}"
ENV LD_LIBRARY_PATH="${NVCOMPILERS}/${NVARCH}/2020/comm_libs/nccl/lib:${LD_LIBRARY_PATH}"
ENV LD_LIBRARY_PATH="${NVCOMPILERS}/${NVARCH}/20.11/comm_libs/nvshmem/lib:${LD_LIBRARY_PATH}"
ENV LD_LIBRARY_PATH="${NVCOMPILERS}/${NVARCH}/2020/comm_libs/nvshmem/lib:${LD_LIBRARY_PATH}"
ENV OPAL_PREFIX="${NVCOMPILERS}/${NVARCH}/20.11/comm_libs/mpi/"
ENV PATH="${NVCOMPILERS}/${NVARCH}/20.11/compilers/bin:${PATH}"
ENV PATH="${NVCOMPILERS}/${NVARCH}/20.11/comm_libs/mpi/bin:${PATH}"
ENV MANPATH="${MANPATH}:${NVCOMPILERS}/${NVARCH}/20.11/comm_libs/mpi/man"

# Install custom compile of boost for mpi.
RUN wget -O ~/boost_1_75_0.tar.bz2 -c 'https://dl.bintray.com/boostorg/release/1.75.0/source/boost_1_75_0.tar.bz2' --max-redirect=100 \
    && apt-get -y remove --purge libboost-all-dev \
    && cd ~ \
    && tar xjf ~/boost_1_75_0.tar.bz2 \
    && cd ~/boost_1_75_0/ \
    && ./bootstrap.sh --prefix=/usr/local --with-toolset=pgi  \
    && cp ~/boost_1_75_0/tools/build/example/user-config.jam ~/user-config.jam \
    && echo "using mpi : mpic++ ; " >> ~/user-config.jam \
    && cd ~/boost_1_75_0/tools/build \
    && ./bootstrap.sh && ./b2 install \
    && cd ~/boost_1_75_0 && b2 toolset=pgi --with-mpi --with-program_options --with-serialization stage install

# Install SuperLU5
RUN cd /opt/code \ 
    && curl -O -J -L https://github.com/xiaoyeli/superlu/archive/v5.2.1.tar.gz \
    && tar xvf superlu-5.2.1.tar.gz \
    && cd superlu-5.2.1 \
    && mkdir build \
    && cd build \
    && cmake ../ -DCMAKE_CXX_COMPILER=nvc++ -DCMAKE_C_COMPILER=nvc -DBUILD_SHARED_LIBS=ON \
    && make \
    && make install

# Install Armadillo
RUN cd opt/code \
    && git clone https://gitlab.com/conradsnicta/armadillo-code \
    && cd armadillo-code \
    && git checkout --track origin/9.900.x \
    && mkdir build \
    && cd build \
    && echo $PATH \
    && cmake ../ -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_CXX_COMPILER=nvc++ -DCMAKE_C_COMPILER=nvc \
    && make \
    && make install \
    && ldconfig

# Install Pybind11 (min required 2.6.2 which brings support for HPC-SDK compilers)
# and install PowerGrid

RUN  cd /opt/code \
     && git clone https://github.com/mavel101/PowerGrid \
     && cd PowerGrid \
     && git checkout icefire \
     && git submodule update --init --recursive \
     && cd pybind11 \
     && mkdir build \
     && cd build \
     && cmake ../ \
     && make \
     && make install \
     && cd /opt/code/PowerGrid \
     && pip3 wheel .

# ----- Start another clean build without all of the build dependencies of siemens_to_ismrmrd -----
FROM ubuntu:20.04
LABEL maintainer="Philipp Ehses (philipp.ehses@dzne.de)"

ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Copy packages from last stage and re-add necessary dependencies
COPY --from=mrd_converter /usr/local/. /usr/local/
COPY --from=mrd_converter /opt/nvidia /opt/nvidia 
COPY --from=mrd_converter /opt/code/spiraltraj/SpiralTraj* /opt/code/
COPY --from=mrd_converter /opt/code/PowerGrid/PowerGridPy* /opt/code/

# Copy MKL
COPY --from=mrd_converter /opt/intel/mkl/lib /opt/intel/mkl/lib/
RUN update-alternatives --install /usr/lib/x86_64-linux-gnu/libblas.so  \
    libblas.so-x86_64-linux-gnu      /opt/intel/mkl/lib/intel64/libmkl_rt.so 50 && \
  update-alternatives --install /usr/lib/x86_64-linux-gnu/libblas.so.3  \
    libblas.so.3-x86_64-linux-gnu    /opt/intel/mkl/lib/intel64/libmkl_rt.so 50 && \
  update-alternatives --install /usr/lib/x86_64-linux-gnu/liblapack.so   \
    liblapack.so-x86_64-linux-gnu    /opt/intel/mkl/lib/intel64/libmkl_rt.so 50 && \
  update-alternatives --install /usr/lib/x86_64-linux-gnu/liblapack.so.3 \
    liblapack.so.3-x86_64-linux-gnu  /opt/intel/mkl/lib/intel64/libmkl_rt.so 50 && \
  echo "/opt/intel/lib/intel64"     >  /etc/ld.so.conf.d/mkl.conf && \
  echo "/opt/intel/mkl/lib/intel64" >> /etc/ld.so.conf.d/mkl.conf && \
  ldconfig && \
  echo "MKL_THREADING_LAYER=GNU" >> /etc/environment

# create pythonpath for single file packages (without requiring setup.py or __init__.py)
RUN mkdir -p /opt/code/pythonpath
ENV PYTHONPATH "${PYTHONPATH}:/opt/code/pythonpath"

# copy bart python interface to pythonpath
COPY --from=mrd_converter /opt/code/bart/python/* /opt/code/pythonpath/

# Dependencies for Python MRD server
RUN  apt-get update --quiet \
     && apt-get dist-upgrade --yes \
     && apt-get install --no-install-recommends --no-install-suggests --yes \
        ca-certificates nano git gcc python3 python3-pip h5utils libxslt1.1 libfftw3-3 liblapacke openmpi-bin libhdf5-dev libatomic1\
     && pip3 install --no-cache-dir pyxb h5py intel-openmp

RUN  mkdir -p /opt/code

# PowerGrid
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.8/dist-packages:${LD_LIBRARY_PATH}"
RUN  cd /opt/code \
     && pip3 install PowerGridPy* \
     && rm PowerGrid*

RUN  cd /opt/code \
     && pip3 install SpiralTraj* \
     && rm SpiralTraj*

RUN  cd /opt/code \
     && git clone https://github.com/ismrmrd/ismrmrd-python.git \
     && cd /opt/code/ismrmrd-python \
     && git checkout 80fecd0 \
     && python3 setup.py install

RUN  cd /opt/code \
     && git clone https://github.com/ismrmrd/ismrmrd-python-tools.git \
     && cd /opt/code/ismrmrd-python-tools \
     && python3 setup.py install

RUN  cd /opt/code \
     && git clone -b bart https://github.com/pehses/python-ismrmrd-server.git

# Cleanup files not required after compiling
RUN  apt-get remove --yes gcc git python3-pip
RUN  apt-get autoremove --yes 
RUN  apt-get autoclean --yes
RUN  rm -r /root/.cache/pip

# add ismrmrd server start script
RUN echo "/opt/code/python-ismrmrd-server/main.py -v -H=0.0.0.0 -p=9002 -l=/tmp/share/debug/python-ismrmrd-server.log" >> /usr/bin/start_server
RUN chmod +x /usr/bin/start_server


CMD ["/usr/bin/start_server"]
