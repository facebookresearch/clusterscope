# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from clusterscope.cluster_info import ClusterInfo

cluster_info = ClusterInfo()


def cluster() -> str:
    return cluster_info.get_cluster_name()
