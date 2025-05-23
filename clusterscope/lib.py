from clusterscope.cluster_info import ClusterInfo

cluster_info = ClusterInfo()


def cluster() -> str:
    return cluster_info.get_cluster_name()
