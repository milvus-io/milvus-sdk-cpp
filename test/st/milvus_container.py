#!/usr/bin/env python3
"""
Milvus standalone container for integration testing.

Usage:
    pip install docker
    python milvus_container.py start    # Start container, prints container ID
    python milvus_container.py stop <container_id>  # Stop container by ID
"""

import argparse
import sys
import time

import docker
import requests

milvus_image = "milvusdb/milvus:v2.6.14"
milvus_port = 19510


def wait_for_milvus_ready(host, port, timeout):
    """
    Wait for Milvus to be fully initialized by calling its REST API.
    """
    url = f"http://{host}:{port}/v2/vectordb/collections/list"
    headers = {
        "Authorization": "Bearer root:Milvus",
        "Content-Type": "application/json",
    }
    payload = {"dbName": "_default"}

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=5)
            if resp.status_code == 200:
                return True
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(1)

    raise TimeoutError(f"Timeout waiting for Milvus to be ready at {url}")


def start_milvus_container(port: int = milvus_port, image: str = milvus_image, timeout: int = 60):
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

    # Wait for Milvus to be fully initialized via REST API
    try:
        wait_for_milvus_ready("localhost", port, timeout=timeout)
    except TimeoutError:
        # Print container logs to help debug
        print("Container logs:", file=sys.stderr)
        print(container.logs().decode("utf-8", errors="replace"), file=sys.stderr)
        container.stop()
        container.remove()
        raise

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
