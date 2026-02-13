#!/usr/bin/env python3
"""
Milvus standalone container for integration testing.

Usage:
    pip install docker
    python milvus_container.py start    # Start container, prints container ID
    python milvus_container.py stop <container_id>  # Stop container by ID
"""

import argparse
import socket
import sys
import time

import docker

milvus_image = "milvusdb/milvus:v2.6.9"
milvus_port = 19530


def wait_for_container_port(container, port, timeout):
    """
    Wait for a port to be ready inside the container using docker exec.
    """
    start_time = time.time()

    while time.time() - start_time < timeout:
        exit_code, _ = container.exec_run(f"bash -c 'echo > /dev/tcp/127.0.0.1/{port}'")
        if exit_code == 0:
            return True
        time.sleep(1)

    raise TimeoutError(f"Timeout waiting for port {port} inside container")


def start_milvus_container(port: int = milvus_port, image: str = milvus_image, timeout: int = 180):
    """
    Start a Milvus standalone container for testing.

    Args:
        port: Host port to bind Milvus gRPC service (default: 19530)
        image: Milvus Docker image to use
        timeout: Timeout in seconds to wait for port (default: 180)

    Returns:
        str: The container ID
    """
    client = docker.from_env()

    # Pull image if not exists
    try:
        client.images.get(image)
    except docker.errors.ImageNotFound:
        try:
            client.images.pull(image)
        except docker.errors.APIError as e:
            raise RuntimeError(f"Failed to pull image {image}: {e}") from e

    container = client.containers.run(
        image,
        command="milvus run standalone",
        environment={
            "ETCD_USE_EMBED": "true",
            "ETCD_DATA_DIR": "/var/lib/milvus/etcd",
            "COMMON_STORAGETYPE": "local",
            "DEPLOY_MODE": "STANDALONE",
        },
        ports={"19530/tcp": port},
        detach=True,
    )

    # Wait for Milvus gRPC port to be ready inside the container
    try:
        wait_for_container_port(container, port, timeout=timeout)
    except TimeoutError:
        # Print container logs to help debug
        print("Container logs:", file=sys.stderr)
        print(container.logs().decode("utf-8", errors="replace"), file=sys.stderr)
        container.stop()
        container.remove()
        raise

    # Additional wait for Milvus to fully initialize after port is open
    time.sleep(10)

    return container.id


def stop_milvus_container(container_id: str):
    """Stop the Milvus container by ID."""
    client = docker.from_env()
    try:
        container = client.containers.get(container_id)
        container.stop()
        container.remove()
        print(f"Milvus container {container_id[:12]} stopped", file=sys.stderr)
    except docker.errors.NotFound:
        print(f"Container {container_id[:12]} not found", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Milvus container management for testing")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Start command
    start_parser = subparsers.add_parser("start", help="Start Milvus container")
    start_parser.add_argument("--port", type=int, default=milvus_port, help=f"Port to bind (default: {milvus_port})")
    start_parser.add_argument("--image", type=str, default=milvus_image, help=f"Milvus image (default: {milvus_image})")

    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop Milvus container")
    stop_parser.add_argument("container_id", help="Container ID to stop")

    args = parser.parse_args()

    if args.command == "start":
        container_id = start_milvus_container(port=args.port, image=args.image)
        # Print container ID to stdout for capture by C++ code
        print(container_id)
    elif args.command == "stop":
        stop_milvus_container(args.container_id)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
