from flask import Flask, request, jsonify, Response
import requests
import time
import psutil
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)

REQUEST_COUNT = Counter("http_requests_total", "Total HTTP Requests")
REQUEST_LATENCY = Histogram("http_request_duration_seconds", "HTTP Request Latency")
THROUGHPUT = Counter("http_requests_throughput", "Total number of requests per second")
CPU_USAGE = Gauge("system_cpu_usage", "CPU Usage Percentage")
RAM_USAGE = Gauge("system_ram_usage", "RAM Usage Percentage")

REQUEST_SUCCESS = Counter("http_requests_success_total", "Total Successful HTTP Requests")
REQUEST_FAILED = Counter("http_requests_failed_total", "Total Failed HTTP Requests")
DISK_USAGE = Gauge("system_disk_usage_percent", "Disk Usage Percentage on Drive C")
NET_BYTES_SENT = Counter("system_network_bytes_sent_total", "Total Network Bytes Sent")
NET_BYTES_RECV = Counter("system_network_bytes_recv_total", "Total Network Bytes Received")

@app.route("/", methods=["GET"])
def home():
    return "Hello, this is the home page of the Prometheus Exporter!"

@app.route("/metrics", methods=["GET"])
def metrics():
    CPU_USAGE.set(psutil.cpu_percent(interval=0.5))
    RAM_USAGE.set(psutil.virtual_memory().percent)
    
    DISK_USAGE.set(psutil.disk_usage('C:\\').percent)
    
    net_io = psutil.net_io_counters()
    NET_BYTES_SENT.inc(net_io.bytes_sent - NET_BYTES_SENT._value.get())
    NET_BYTES_RECV.inc(net_io.bytes_recv - NET_BYTES_RECV._value.get())

    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()
    REQUEST_COUNT.inc()
    THROUGHPUT.inc()

    api_url = "http://127.0.0.1:5001/invocations"
    data = request.get_json()

    try:
        response = requests.post(api_url, json=data)
        duration = time.time() - start_time
        REQUEST_LATENCY.observe(duration)
        
        # [BARU] Tambah hitungan jika request ke Docker sukses
        REQUEST_SUCCESS.inc()

        return jsonify(response.json())
    
    except Exception as e:
        # [BARU] Tambah hitungan jika request ke Docker gagal/error
        REQUEST_FAILED.inc()
        return jsonify({"error" : str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)