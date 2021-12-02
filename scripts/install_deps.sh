#!/bin/sh

get_cmake_version() {
    if which cmake 2>/dev/null 1>/dev/null ; then
        expr $(cmake --version | grep version | sed 's/\./ /g' | awk '{printf "%02d%02d", $3, $4}')
    else
        echo 0
    fi
}

install_deps_for_ubuntu_1804() {
    sudo apt-get update
    sudo apt-get -y install python2.7 gpg wget gcc g++ ccache clang-format-10 clang-tidy-10
    sudo apt-get -y install libssl-dev iwyu

    # for cmake >= 3.12
    current_cmake_version=$(get_cmake_version)
    if [ $current_cmake_version -lt 312 ] ; then
        sudo rm -f /usr/share/keyrings/kitware-archive-keyring.gpg /etc/apt/sources.list.d/kitware.list
        wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
        echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ bionic main' | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null
        sudo apt-get update
        sudo apt-get -y install kitware-archive-keyring
        sudo apt-get -y install cmake
    fi
}


if [ -x "$(command -v apt)" ]; then
    # for ubuntu
    if lsb_release -r | grep -q 18.04 ; then
        install_deps_for_ubuntu_1804
    fi
elif [ -x "$(command -v yum)" ] ; then
    # todo handle for centos7
    :
fi
