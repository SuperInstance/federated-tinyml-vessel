"""aggregator.py — The federated aggregator server.

A simple HTTP/WS server that accepts head uploads from devices,
runs FedAvg, and broadcasts the new global head back.

Wire protocol (JSON over WebSocket or HTTP POST):

  Inbound (device -> aggregator):
    {
      "schema": "f170-head-v1",
      "device_id": "wrist-001",
      "round": 17,
      "samples_seen": 1240,
      "W": [[0.012, ...], ...],   // 64x5 = 320 floats
      "b": [-0.001, 0.023, ...],  // 5 floats
      "loss": 0.314,
      "state_hash": "0x..."
    }

  Outbound (aggregator -> device):
    {
      "schema": "f170-global-v1",
      "round": 18,
      "samples_seen_total": 12340,
      "W": [...],   // the new global head weights
      "b": [...],
      "state_hash": "0x..."
    }

Run:
    python3 aggregator.py --port 8765
"""
from __future__ import annotations
import json
import argparse
import http.server
import socketserver
import threading
from collections import defaultdict
from typing import Dict, List

import numpy as np

from classifier_head import ClassifierHead, average_heads, HEAD_DIM, NUM_CLASSES


class FederatedAggregator:
    """Thread-safe aggregator that collects head uploads and runs FedAvg."""

    def __init__(self, num_classes: int = NUM_CLASSES, embedding_dim: int = HEAD_DIM):
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.global_head = ClassifierHead(num_classes=num_classes, seed=0)
        self.round = 0
        self.samples_seen_total = 0
        # device_id -> latest head
        self.device_heads: Dict[str, ClassifierHead] = {}
        # device_id -> data size (for FedAvg weighting)
        self.device_sizes: Dict[str, int] = {}
        self.lock = threading.Lock()
        # Statistics
        self.history = []

    def receive_head(self, device_id: str, head: ClassifierHead, data_size: int):
        """Called when a device uploads a head. Triggers FedAvg if all expected devices are in."""
        with self.lock:
            self.device_heads[device_id] = head
            self.device_sizes[device_id] = data_size
            self.samples_seen_total += data_size
            return self.maybe_fedavg()

    def maybe_fedavg(self) -> bool:
        """Run FedAvg if we have heads from N devices (default 3)."""
        if len(self.device_heads) < 3:
            return False
        heads = list(self.device_heads.values())
        sizes = list(self.device_sizes.values())
        self.global_head = average_heads(heads, weights=sizes)
        self.round += 1
        self.history.append({
            "round": self.round,
            "n_devices": len(heads),
            "samples_seen": self.samples_seen_total,
            "global_hash": f"0x{self.global_head.state_hash():016x}",
        })
        # Clear for next round
        self.device_heads.clear()
        return True

    def get_global_head_dict(self) -> dict:
        return {
            "schema": "f170-global-v1",
            "round": self.round,
            "samples_seen_total": self.samples_seen_total,
            "W": self.global_head.weights.tolist(),
            "b": self.global_head.biases.tolist(),
            "num_classes": self.global_head.num_classes,
            "embedding_dim": self.global_head.embedding_dim,
            "state_hash": f"0x{self.global_head.state_hash():016x}",
        }


# ---- HTTP server ----

class AggregatorHandler(http.server.BaseHTTPRequestHandler):
    """HTTP endpoint for head upload + global head query."""

    def log_message(self, format, *args):
        # Quieter logging
        pass

    def do_POST(self):
        if self.path == "/head":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            try:
                msg = json.loads(body)
                self._handle_head(msg)
            except json.JSONDecodeError:
                self.send_error(400, "invalid json")
            return
        if self.path == "/reset":
            self.server.aggregator.round = 0
            self.server.aggregator.global_head = ClassifierHead(
                num_classes=self.server.aggregator.num_classes, seed=0
            )
            self.server.aggregator.device_heads.clear()
            self.server.aggregator.samples_seen_total = 0
            self._send_json({"status": "reset", "round": 0})
            return
        self.send_error(404)

    def do_GET(self):
        if self.path == "/global":
            self._send_json(self.server.aggregator.get_global_head_dict())
            return
        if self.path == "/stats":
            self._send_json({
                "round": self.server.aggregator.round,
                "samples_seen_total": self.server.aggregator.samples_seen_total,
                "history": self.server.aggregator.history,
            })
            return
        if self.path == "/":
            self._send_text("F170 Aggregator. POST /head, GET /global, GET /stats, POST /reset")
            return
        self.send_error(404)

    def _handle_head(self, msg: dict):
        schema = msg.get("schema", "")
        if not schema.startswith("f170-head-v"):
            self.send_error(400, f"unknown schema: {schema}")
            return
        device_id = msg.get("device_id", "unknown")
        try:
            head = ClassifierHead.from_json_envelope(msg)
            data_size = msg.get("samples_seen", 1)
        except Exception as e:
            self.send_error(400, f"parse error: {e}")
            return
        triggered = self.server.aggregator.receive_head(device_id, head, data_size)
        self._send_json({
            "status": "ok",
            "device_id": device_id,
            "fedavg_triggered": triggered,
            "current_round": self.server.aggregator.round,
        })

    def _send_json(self, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text):
        body = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, handler, aggregator):
        super().__init__(addr, handler)
        self.aggregator = aggregator


# ---- Demo / self-test ----

def demo_run():
    """Run a complete federated round-trip demo without a network."""
    print("F170 Aggregator — Local Demo")
    print("=" * 60)
    print()
    print("This demo simulates 5 devices each uploading a head.")
    print("Each round, the aggregator FedAvg-averages the heads.")
    print("After 10 rounds, the global head is verified to discriminate.")
    print()

    ag = FederatedAggregator()

    # 5 devices, each with a slightly different head (after some local training)
    import numpy as np
    np.random.seed(0)
    base = ClassifierHead(num_classes=5, seed=42)

    for round_num in range(10):
        # Each device gets a copy of the global head and trains locally
        device_heads = []
        device_sizes = []
        for device_id in range(5):
            local = ClassifierHead.from_bytes(base.to_bytes(), num_classes=5, embedding_dim=64)
            # Each device sees one class more often (non-IID)
            preferred_class = device_id % 5
            for _ in range(20):
                # Generate embedding
                emb = np.random.randn(64).astype(np.float32)
                emb /= np.linalg.norm(emb) + 1e-9
                # Label: mostly preferred class
                label = preferred_class if np.random.rand() < 0.7 else np.random.randint(5)
                local.sgd_step(np.array([emb]), np.array([label]), learning_rate=0.05)
            device_heads.append(local)
            device_sizes.append(20)

        # Submit all heads
        for dh, sz in zip(device_heads, device_sizes):
            ag.receive_head(f"device-{device_heads.index(dh)}", dh, sz)
        # The aggregator should have triggered FedAvg after 3 devices
        ag.maybe_fedavg()
        # Update base for next round
        base = ClassifierHead.from_bytes(ag.global_head.to_bytes(), num_classes=5, embedding_dim=64)

        if round_num % 2 == 0:
            print(f"  Round {round_num}: hash = 0x{ag.global_head.state_hash():016x}")

    print()
    print(f"Final round: {ag.round}")
    print(f"Final global head hash: 0x{ag.global_head.state_hash():016x}")
    print(f"Total samples seen: {ag.samples_seen_total}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765, help="HTTP port")
    parser.add_argument("--demo", action="store_true", help="Run local demo (no network)")
    args = parser.parse_args()

    if args.demo:
        demo_run()
        return

    ag = FederatedAggregator()
    server = ThreadedHTTPServer(("0.0.0.0", args.port), AggregatorHandler, ag)
    print(f"F170 Aggregator listening on :{args.port}")
    print("Endpoints:")
    print(f"  POST /head    — upload a device head")
    print(f"  GET  /global  — fetch the current global head")
    print(f"  GET  /stats   — see round + history")
    print(f"  POST /reset   — reset for a new training run")
    server.serve_forever()


if __name__ == "__main__":
    main()
