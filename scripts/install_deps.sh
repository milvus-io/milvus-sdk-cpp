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
                       libssl-dev iwyu lcov git

    # for cmake >= 3.12
    current_cmake_version=$(get_cmake_version)
    if [ $current_cmake_version -lt 312 ] ; then
        ${SUDO} rm -f /usr/share/keyrings/kitware-archive-keyring.gpg /etc/apt/sources.list.d/kitware.list
        wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | ${SUDO} tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
        echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ ${dist} main" | ${SUDO} tee /etc/apt/sources.list.d/kitware.list >/dev/null
        ${SUDO} apt-get update
        ${SUDO} apt-get -y install kitware-archive-keyring
        ${SUDO} apt-get -y install cmake
    fi

    # install new clang-tidy clang-format
    wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | ${SUDO} apt-key add -
    ${SUDO} apt-get -y install software-properties-common
    ${SUDO} add-apt-repository "deb http://apt.llvm.org/${dist}/   llvm-toolchain-${dist}-13  main"
    ${SUDO} apt-get update
    ${SUDO} apt-get install -y clang-format-13 clang-tidy-13
}

install_deps_for_ubuntu_1804() {
    install_deps_for_ubuntu_common bionic
}

install_deps_for_ubuntu_2004() {
    install_deps_for_ubuntu_common focal
}

install_deps_for_fedora_common() {
    check_sudo
    ${SUDO} dnf -y install gcc gcc-c++ python2 gpg wget ccache make openssl-devel which lcov git rpm-build
    ${SUDO} dnf -y install cmake
}

install_deps_for_centos_8() {
    check_sudo
    ${SUDO} dnf -y install epel-release
    ${SUDO} dnf -y install gcc gcc-c++ python2 gpg wget ccache make openssl-devel which lcov git rpm-build
    ${SUDO} dnf -y install cmake
}

install_deps_for_centos_7() {
    check_sudo
    ${SUDO} yum -y install epel-release
    ${SUDO} yum -y install gcc gcc-c++ python gpg wget ccache make openssl-devel which lcov git rpm-build

    # for cmake >= 3.12, using cmake3 from epel
    current_cmake_version=$(get_cmake_version)
    if [ $current_cmake_version -lt 312 ] ; then
        ${SUDO} yum -y install cmake3
        test -L /usr/local/bin/cmake && ${SUDO} unlink /usr/local/bin/cmake
        ${SUDO} ln -s /usr/bin/cmake3 /usr/local/bin/cmake
    fi
}

install_deps_for_macos() {
    if [ -x "$(command -v brew)" ] ; then
        brew install wget lcov llvm openssl@3 cmake ccache
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
        elif grep -q 'Fedora release 34' /etc/redhat-release ; then
            install_deps_for_fedora_common
        elif grep -q 'Fedora release 35' /etc/redhat-release ; then
            install_deps_for_fedora_common
        fi
    fi
elif uname | grep -wq Darwin ; then
    install_deps_for_macos
fi

