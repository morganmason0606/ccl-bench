def metric_cal(directory: str) -> float:
    """
    Calculate the bandwidth utilization from the exported sqlite file from nsys.

    Args:
        directory (str): The directory path containing the exported sqlite file from nsys.

    Returns:
        float: The calculated bandwidth utilization value.
    """

    """
    allgather csv has lots of misc data in it, bandwidth_utilization is the headliner
    - top prioirty is getting the mediabn
    - also nice to have p25, p75, p99, (mean, std)

    note that allgather has only been calcualted for tp>1 and for the last stage of pp when pp>1

    n/a for llama tp=1 and qwen pp=2,node 0
    """
    allgather = {
        
        # "total": 0.0, no meaningful 'total' 
        "mean": 0.0,
        "median": 0.0,
        "std": 0.0,
        "p99": 0.0,
    }
    """
    all reduce is like allgather
    - csv has lots of misc data in it, bandwidth_utilization is the headliner
    - top prioirty is getting the mediabn
    - also nice to have p25, p75, p99, (mean, std)

    na for just llama tp=1; still applicable to both qwens
    """
    allreduce = {
    }
    

    """
    a2a was !!!only collected for deepseek!, and is only really applicable to ep>1, but we also collected info about ep-1 for a baseline because the code is kind of jank 
    - there are multiple types of nvlink communication, right now just get NVLink TX Responses User Data; NVLink RX Responses UserData
        - min, max, average
        - and if you have more space, get everything 

    only applicable for deepseek ep>1
    """

    a2a = {}


    """
    p2p like a2a was only collected pp>1
    - we also collected for both sides of qwen pp=2, node=0,1

    - there are multiple types of nvlink communication, right now just get NVLink TX Responses User Data; NVLink RX Responses UserData
        - min, max, average
        - and if you have more space, get everything 
    - also want to collect PCIe Rx, Tx; especially for qwen pp=2, nodes = 2
    
    only applicable pp>1: llama pp>1, qwen pp=2 both nodes
    """
    p2p = {}
    return {
        "AllGather": allgather,
        "AllReduce": allreduce,
        "AllToAll": a2a,
        "PeerToPeer": p2p,
    }