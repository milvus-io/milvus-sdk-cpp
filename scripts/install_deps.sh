#!/bin/sh

check_sudo() {
    if sudo -V 2>/dev/null ; then
        export SUDO=sudo
    else
        export SUDO=
    fi
}

install_linux_cmake_clang_toolchain() {
    pip3 install --user -U pip scikit-build wheel
    pip3 install --user cmake clang-tidy~=17.0 clang-format~=17.0
}

install_deps_for_ubuntu_common() {
    dist=$1
    check_sudo

    ${SUDO} apt-get update

    # patch for install tzdata under noninteractive
    if [ -z "${SUDO}" ] ; then
        DEBIAN_FRONTEND=noninteractive DEBCONF_NONINTERACTIVE_SEEN=true apt-get -y install tzdata
    else
        ${SUDO} DEBIAN_FRONTEND=noninteractive DEBCONF_NONINTERACTIVE_SEEN=true apt-get -y install tzdata
    fi

    ${SUDO} apt-get -y install python2.7 gpg wget gcc g++ ccache make \
                       libssl-dev iwyu lcov git python3-pip
    install_linux_cmake_clang_toolchain
}

install_deps_for_ubuntu_1804() {
    install_deps_for_ubuntu_common bionic
}

install_deps_for_ubuntu_2004() {
    install_deps_for_ubuntu_common focal
}

install_deps_for_ubuntu_2204() {
    install_deps_for_ubuntu_common jammy
}

install_deps_for_fedora_common() {
    check_sudo
    ${SUDO} dnf -y install gcc gcc-c++ python2 gpg wget ccache make openssl-devel which lcov git rpm-build python3-pip
    install_linux_cmake_clang_toolchain
}

install_deps_for_centos_8() {
    check_sudo
    ${SUDO} dnf -y install epel-release
    ${SUDO} dnf -y install gcc gcc-c++ python2 gpg wget ccache make openssl-devel which lcov git rpm-build python3-pip
    install_linux_cmake_clang_toolchain
}

install_deps_for_centos_7() {
    check_sudo
    ${SUDO} yum -y install epel-release centos-release-scl
    ${SUDO} yum -y install gcc gcc-c++ python gpg wget ccache make openssl-devel which lcov git rpm-build python3-pip
    ${SUDO} yum -y install devtoolset-7

    scl enable devtoolset-7 bash
    install_linux_cmake_clang_toolchain
}

install_deps_for_macos() {
    if [ -x "$(command -v brew)" ] ; then
        brew install wget lcov llvm openssl@3 ccache
    else
        echo 'Detect using macos but brew seems not installed.'
        exit 1
    fi
    install_linux_cmake_clang_toolchain
}

if uname | grep -wq Linux ; then
    if [ -x "$(command -v apt)" ]; then
        # for ubuntu
        if grep -q 'Ubuntu 18.04' /etc/issue ; then
            install_deps_for_ubuntu_1804
        elif grep -q 'Ubuntu 20.04' /etc/issue ; then
            install_deps_for_ubuntu_2004
        elif grep -q 'Ubuntu 22.04' /etc/issue ; then
            install_deps_for_ubuntu_2204
        fi
    elif [ -x "$(command -v yum)" ] ; then
        # for os support yum
        if grep -q 'CentOS Linux release 7' /etc/redhat-release ; then
            install_deps_for_centos_7
        elif grep -q 'Red Hat Enterprise Linux release 7' /etc/redhat-release ; then
            install_deps_for_centos_7
        elif grep -q 'CentOS Linux release 8' /etc/redhat-release ; then
            install_deps_for_centos_8
        elif grep -q 'CentOS Stream release 8' /etc/redhat-release ; then
            install_deps_for_centos_8
        elif grep -q 'Red Hat Enterprise Linux release 8' /etc/redhat-release ; then
            install_deps_for_centos_8
        elif grep -q 'Fedora release' /etc/redhat-release ; then
            install_deps_for_fedora_common
        fi
    fi
elif uname | grep -wq Darwin ; then
    install_deps_for_macos
fi

