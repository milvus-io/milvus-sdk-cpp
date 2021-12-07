#!/bin/sh

check_sudo() {
    if [ -x "$(command -v sudo)" ] ; then
        export SUDO=sudo
    else
        export SUDO=
    fi
}

get_cmake_version() {
    if [ -x "$(command -v cmake)" ] ; then
        expr $(cmake --version | grep version | sed 's/\./ /g' | awk '{printf "%02d%02d", $3, $4}')
    else
        echo 0
    fi
}

install_deps_for_ubuntu_1804() {
    check_sudo
    ${SUDO} apt-get update
    ${SUDO} apt-get -y install python2.7 gpg wget gcc g++ ccache make clang-format-10 clang-tidy-10 \
                       libssl-dev iwyu

    # for cmake >= 3.12
    current_cmake_version=$(get_cmake_version)
    if [ $current_cmake_version -lt 312 ] ; then
        ${SUDO} rm -f /usr/share/keyrings/kitware-archive-keyring.gpg /etc/apt/sources.list.d/kitware.list
        wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | ${SUDO} tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
        echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ bionic main' | ${SUDO} tee /etc/apt/sources.list.d/kitware.list >/dev/null
        ${SUDO} apt-get update
        ${SUDO} apt-get -y install kitware-archive-keyring
        ${SUDO} apt-get -y install cmake
    fi
}

install_deps_for_centos_7() {
    check_sudo
    ${SUDO} yum -y update
    ${SUDO} yum -y install epel-release
    yum -y install gcc gcc-c++ python gpg wget ccache make openssl-devel which
    
    # for cmake >= 3.12, using cmake3 from epel
    current_cmake_version=$(get_cmake_version)
    if [ $current_cmake_version -lt 312 ] ; then
        ${SUDO} yum -y install cmake3
        test -L /usr/local/bin/cmake && ${SUDO} unlink /usr/local/bin/cmake
        ${SUDO} ln -s /usr/bin/cmake3 /usr/local/bin/cmake
    fi
}


if [ -x "$(command -v apt)" ]; then
    # for ubuntu
    if grep -q 'Ubuntu 18.04' /etc/issue ; then
        install_deps_for_ubuntu_1804
    fi
elif [ -x "$(command -v yum)" ] ; then
    # for os support yum
    if grep -q 'CentOS Linux release 7' /etc/redhat-release ; then
        install_deps_for_centos_7
    fi
fi
