mkdir -p /pscratch/sd/b/bck/kineto_for_tool/llama-3.1-8b-pp-4

i=0
for f in /pscratch/sd/b/bck/kineto_out/llama-3.1-8b-pp-4/*.pt.trace.json.gz; do
    gunzip -c "$f" > /pscratch/sd/b/bck/kineto_for_tool/llama-3.1-8b-pp-4/kineto_trace_${i}.json
    i=$((i+1))
done

ls /pscratch/sd/b/bck/kineto_for_tool/llama-3.1-8b-pp-4/kineto_trace_*.json