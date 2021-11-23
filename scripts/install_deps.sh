#!/bin/sh

install_deps_for_ubuntu() {
    sudo apt -y install cmake ccache clang-format clang-tidy
}


if [ -x "$(command -v apt)" ]; then
    # for ubuntu
    install_deps_for_ubuntu
elif [ -x "$(command -v yum)" ] ; then
    # todo handle for centos7
    :
fi
