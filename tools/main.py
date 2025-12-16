import argparse

if __name__ == "__main__":
    trace_directory = None
    metric_name = None
    metric_cal_func = None

    parser = argparse.ArgumentParser(description="Process trace directory and metric name.")
    parser.add_argument("--trace", type=str, required=True, help="Path to the trace directory")
    parser.add_argument("--metric", type=str, required=True, help="Name of the metric to calculate")

    args = parser.parse_args()

    trace_directory = args.trace
    metric_name = args.metric
    
    if metric_name == "coll_call_num":
        from coll_call_num.coll_call_num import metric_cal
        metric = metric_cal(trace_directory)
        print(metric)
    elif metric_name == "straggler_metrics":
        from straggler.straggler_metrics import metric_cal
        delay, slowdown = metric_cal(trace_directory)
        print("Straggler Delay: ", delay)
        print("Straggler Slowdown: ", slowdown)
    else:
        raise ValueError(f"Unsupported metric name: {metric_name}")
    

