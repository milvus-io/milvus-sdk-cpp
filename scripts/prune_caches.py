#!/usr/bin/env python3
# Licensed to the LF AI & Data foundation under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Best-effort pruning of old GitHub Actions cache generations.

Cache entries are immutable, so every push saves a new key. This script deletes
older generations of the configured prefixes so the repository stays below the
10 GB cache quota instead of relying on GitHub's eviction. It is deliberately
infallible: any API error (including the read-only GITHUB_TOKEN of fork pull
requests) only prints a warning and never fails the build.

Environment:
  GH_TOKEN         token for the GitHub API
  PRUNE_PREFIXES   space-separated key prefixes to prune
  KEEP_KEYS        space-separated keys that must survive
"""

import json
import os
import urllib.request


def call_api(path, token, method="GET"):
    request = urllib.request.Request(
        "https://api.github.com" + path,
        headers={"Authorization": "Bearer " + token, "Accept": "application/vnd.github+json"},
        method=method,
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response) if response.status != 204 else None


def main():
    token = os.environ.get("GH_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")
    ref = os.environ.get("GITHUB_REF")
    if not token or not repo or not ref:
        print("WARN: GH_TOKEN/GITHUB_REPOSITORY/GITHUB_REF are not all set, skipping prune")
        return
    prefixes = os.environ.get("PRUNE_PREFIXES", "").split()
    keeps = set(os.environ.get("KEEP_KEYS", "").split())

    page = 1
    total = None
    entries = []
    while True:
        try:
            result = call_api("/repos/%s/actions/caches?ref=%s&per_page=100&page=%d" % (repo, ref, page), token)
        except Exception as error:  # noqa: BLE001 - pruning must never fail the build
            print("WARN: could not list caches (page %d), skipping prune: %s" % (page, error))
            return
        entries.extend(result.get("actions_caches", []))
        if total is None:
            total = result.get("total_count", 0)
        if len(entries) >= total or not result.get("actions_caches"):
            break
        page += 1
        if page > 100:  # hard cap; 10k entries is far beyond any realistic repository
            break

    # Retain the newest non-kept entry per prefix as a fallback so a run cancelled
    # between this prune and its post-job cache save (concurrency cancel-in-progress)
    # cannot take the chain from warm to empty; steady state is at most two
    # generations per prefix.
    fallbacks = set()
    for prefix in prefixes:
        matching = [c for c in entries if c["key"].startswith(prefix) and c["key"] not in keeps]
        if matching:
            newest = max(matching, key=lambda c: c.get("created_at", ""))
            fallbacks.add(newest["key"])

    removed = 0
    failed = 0
    for cache in entries:
        key = cache["key"]
        if any(key.startswith(prefix) for prefix in prefixes) and key not in keeps and key not in fallbacks:
            try:
                call_api("/repos/%s/actions/caches/%d" % (repo, cache["id"]), token, "DELETE")
                removed += 1
            except Exception as error:  # noqa: BLE001 - best effort by design
                failed += 1
                print("WARN: could not prune %s: %s" % (key, error))
    print("Pruned %d old cache generation(s), %d could not be pruned" % (removed, failed))


if __name__ == "__main__":
    main()
