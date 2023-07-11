FROM nvcr.io/nvidia/pytorch:22.01-py3

MAINTAINER Kazuki Iwasaki <k897706@kansai-u.ac.jp>

ENV https_proxy http://proxy.itc.kansai-u.ac.jp:8080
ENV http_proxy http://proxy.itc.kansai-u.ac.jp:8080

# Specify number of CPUs can be used while building Tensorflow and OpenCV.
ARG NUM_CPUS_FOR_BUILD=4
RUN echo 'Acquire::http::proxy "http://proxy.itc.kansai-u.ac.jp:8080/";\nAcquire::http::proxy "http://proxy.itc.kansai-u.ac.jp:8080/";\nAcquire::ftp::proxy "http://proxy.itc.kansai-u.ac.jp:8080/";\nAcquire::socks::proxy "socks://proxy.itc.kansai-u.ac.jp:8080/";' >> /etc/apt/apt.conf
RUN echo 'http_proxy="http://proxy.itc.kansai-u.ac.jp:8080/"\nhttps_proxy="http://proxy.itc.kansai-u.ac.jp:8080/"\nftp_proxy="http://proxy.itc.kansai-u.ac.jp:8080/"\nsocks_proxy="http://proxy.itc.kansai-u.ac.jp:8080/"' >> /etc/environment
RUN echo 'https_proxy = http://proxy.itc.kansai-u.ac.jp:8080/\nhttp_proxy = http://proxy.itc.kansai-u.ac.jp:8080/\nftp_proxy = http://proxy.itc.kansai-u.ac.jp:8080/' >>/etc/wgetrc
RUN sed -i.bak -e "s%http://archive.ubuntu.com/ubuntu/%http://ftp.jaist.ac.jp/pub/Linux/ubuntu/%g" /etc/apt/sources.list

# Time zone settings
RUN sed -i.bak -e "s%http://archive.ubuntu.com/ubuntu/%http://ftp.jaist.ac.jp/pub/Linux/ubuntu/%g" /etc/apt/sources.list
ENV TZ Asia/Tokyo
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y \
  && apt-get install -y tzdata \
  && rm -rf /var/lib/apt/lists/* \
  && echo "${TZ}" > /etc/timezone \
  && rm /etc/localtime \
  && ln -s /usr/share/zoneinfo/Asia/Tokyo /etc/localtime \
  && dpkg-reconfigure -f noninteractive tzdata

# Install some useful and machine/deep-learning-related packages for Python3.
RUN mkdir -p /home/src
COPY requirements.txt /home/src
WORKDIR /home/src
#
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install -r ./requirements.txt
#
ENV USER student
ENV HOME /home/${USER}
ENV SHELL /usr/bin/zsh
#
RUN groupadd -g 1002 student
RUN useradd -g 1002 -u 1002 -m -s /bin/bash ${USER}
#
RUN gpasswd -a ${USER} sudo
#
RUN echo "${USER}:student" | chpasswd
#
RUN echo 'Defaults visiblepw'             >> /etc/sudoers
RUN echo '${USER} ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
#
RUN apt-get -y clean all
RUN rm -rf /var/lib/apt/lists/*
RUN apt-get update -y && apt-get install -y \
    zsh \
    git \
    fd-find \
    libfuse2 \
    fuse3 \
    ffmpeg \
    libsm6 \
    libxext6 \
    gcc make \
    pkg-config autoconf automake \
    python3-docutils \
    libseccomp-dev \
    libjansson-dev \
    libyaml-dev \
    libxml2-dev
RUN curl -SL https://deb.nodesource.com/setup_20.x | bash
RUN apt-get install -y nodejs
# neovim
WORKDIR /home/student
RUN curl -LO https://github.com/neovim/neovim/releases/latest/download/nvim.appimage
RUN chmod u+x nvim.appimage
RUN mv nvim.appimage nvim
RUN chown student:student nvim
# ctag
WORKDIR /home/src
RUN git clone https://github.com/universal-ctags/ctags.git
WORKDIR /home/src/ctags
RUN ./autogen.sh
RUN ./configure --prefix=/usr/local
RUN make
RUN make install
# fzf
RUN git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf
RUN ~/.fzf/install
# chown
RUN mkdir ~/.config/
RUN chown student:student ~/.config
#
USER student
RUN zsh
