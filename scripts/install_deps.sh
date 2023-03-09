#!/bin/sh

check_sudo() {
    if sudo -V 2>/dev/null ; then
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
                       libssl-dev iwyu lcov git python3 python3-pip

    # for cmake >= 3.12
    current_cmake_version=$(get_cmake_version)
    if [ $current_cmake_version -lt 312 ] ; then
        pip3 install skbuild
        pip3 install cmake
    fi

    llvm_version=14
    if [ "${dist}" = "bionic" ] ; then
        llvm_version=13
    fi

    # install stable clang-tidy clang-format
    wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | ${SUDO} apt-key add -
    ${SUDO} apt-get -y install software-properties-common
    ${SUDO} add-apt-repository -y "deb http://apt.llvm.org/${dist}/   llvm-toolchain-${dist}-${llvm_version}  main"
    ${SUDO} apt-get update
    ${SUDO} apt-get install -y clang-format-${llvm_version} clang-tidy-${llvm_version}

    # install conan
    pip3 install 'conan>=2.0'
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
    ${SUDO} dnf -y install gcc gcc-c++ python2 gpg wget ccache make which lcov git rpm-build
    ${SUDO} dnf -y install cmake python3 python3-pip

    # install conan
    pip3 install 'conan>=2.0'
}

install_deps_for_centos_8() {
    check_sudo
    ${SUDO} dnf -y install epel-release
    ${SUDO} dnf -y install gcc gcc-c++ python2 gpg wget ccache make which lcov git rpm-build
    ${SUDO} dnf -y install cmake python3 python3-pip

    # install conan
    pip3 install 'conan>=2.0'
}

install_deps_for_centos_7() {
    check_sudo
    ${SUDO} yum -y install epel-release centos-release-scl
    ${SUDO} yum -y install gcc gcc-c++ python gpg wget ccache make which lcov git rpm-build
    ${SUDO} yum -y install devtoolset-7 python36 python36-pip

    # install conan, cmake
    pip3 install skbuild
    pip3 install 'conan>=2.0' cmake

    scl enable devtoolset-7 bash
}

install_deps_for_macos() {
    if [ -x "$(command -v brew)" ] ; then
        brew install wget lcov llvm cmake ccache conan
    else
        echo 'Detect using macos but brew seems not installed.'
        exit 1
    fi
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

