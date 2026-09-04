"""device_client.py — The on-device federated learning client.

This is the code that runs on each vessel device (wrist, deck, bridge, etc.).
It:

1. Fetches the current global head from the aggregator (or starts with a fresh one)
2. Runs local SGD on the device's own audio data
3. Uploads the updated head to the aggregator
4. Waits for the next round

The "data" here is synthetic — in production this would be the device's
actual audio stream, classified via the frozen backbone, then used to
update the head.

Wire protocol: HTTP POST to the aggregator's /head endpoint.
"""
from __future__ import annotations
import argparse
import json
import time
import sys
import numpy as np
import urllib.request
import urllib.error

from classifier_head import ClassifierHead, NUM_CLASSES, HEAD_DIM
from simulator import generate_sample


def http_post(url, data, retries=3):
    """POST JSON data, with retries."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"),
                                          headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, ConnectionError) as e:
            if attempt == retries - 1:
                raise
            time.sleep(0.5 * (attempt + 1))


def http_get(url):
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


class DeviceClient:
    """A simulated vessel device (wrist / deck / bridge / etc.)."""

    def __init__(self, device_id: str, aggregator_url: str,
                 preferred_class: int = 0,
                 num_classes: int = NUM_CLASSES,
                 embedding_dim: int = HEAD_DIM):
        self.device_id = device_id
        self.aggregator_url = aggregator_url.rstrip("/")
        self.preferred_class = preferred_class
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        # Local data generator
        self.rng = np.random.default_rng(hash(device_id) & 0xFFFFFFFF)
        # Local head (will be replaced with global head on first fetch)
        self.local_head = ClassifierHead(num_classes=num_classes, embedding_dim=embedding_dim, seed=0)
        self.local_samples = 0

    def fetch_global_head(self) -> ClassifierHead:
        """Pull the current global head from the aggregator."""
        try:
            data = http_get(f"{self.aggregator_url}/global")
        except Exception as e:
            print(f"  [{self.device_id}] failed to fetch global: {e}")
            return self.local_head
        h = ClassifierHead(num_classes=data["num_classes"], embedding_dim=data["embedding_dim"])
        h.weights = np.array(data["W"], dtype=np.float32)
        h.biases = np.array(data["b"], dtype=np.float32)
        h.steps = 0
        h.samples_seen = 0
        return h

    def generate_local_batch(self, batch_size: int = 20):
        """Generate a non-IID batch of (embedding, label) from the simulator.

        In production, this would be real audio -> backbone -> embedding.
        For the demo, we generate embeddings directly that are class-discriminative.
        """
        embeddings = []
        labels = []
        for _ in range(batch_size):
            # 70% chance of preferred class (non-IID)
            if self.rng.random() < 0.7:
                label = self.preferred_class
            else:
                label = self.rng.integers(0, self.num_classes)
            # Generate a class-discriminative embedding
            emb = self.rng.standard_normal(self.embedding_dim).astype(np.float32) * 0.3
            # Strong class signal
            emb[label * 12:(label + 1) * 12] += 2.0
            # Normalize
            emb /= np.linalg.norm(emb) + 1e-9
            embeddings.append(emb)
            labels.append(label)
        return np.stack(embeddings), np.array(labels, dtype=np.int64)

    def run_local_sgd(self, local_steps: int = 5, batch_size: int = 8, lr: float = 0.05) -> float:
        """Run N local SGD steps on the device's data."""
        total_loss = 0.0
        for step in range(local_steps):
            x, y = self.generate_local_batch(batch_size=batch_size)
            loss = self.local_head.sgd_step(x, y, learning_rate=lr)
            total_loss += loss
            self.local_samples += batch_size
        return total_loss / local_steps

    def upload(self) -> dict:
        """Ship the head to the aggregator."""
        env = json.loads(self.local_head.to_json_envelope(
            round=0, device_id=self.device_id, samples_seen=self.local_samples
        ))
        try:
            return http_post(f"{self.aggregator_url}/head", env)
        except Exception as e:
            return {"status": "error", "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", default="wrist-001", help="unique device id")
    parser.add_argument("--preferred-class", type=int, default=0, help="the class this device hears most")
    parser.add_argument("--aggregator", default="http://localhost:8766", help="aggregator URL")
    parser.add_argument("--rounds", type=int, default=10, help="how many rounds to participate in")
    parser.add_argument("--local-steps", type=int, default=5, help="local SGD steps per round")
    parser.add_argument("--lr", type=float, default=0.05)
    args = parser.parse_args()

    client = DeviceClient(args.device_id, args.aggregator,
                          preferred_class=args.preferred_class)
    print(f"[{args.device_id}] starting; preferred_class={args.preferred_class}, "
          f"aggregator={args.aggregator}, rounds={args.rounds}")

    for round_idx in range(args.rounds):
        # Fetch the current global head
        client.local_head = client.fetch_global_head()
        # Run local training
        loss = client.run_local_sgd(local_steps=args.local_steps, lr=args.lr)
        # Upload
        result = client.upload()
        # Sleep a bit between rounds
        time.sleep(0.1)
        if result.get("status") == "ok":
            print(f"  round {round_idx}: loss={loss:.4f}, "
                  f"upload OK, fedavg_triggered={result.get('fedavg_triggered')}, "
                  f"current_round={result.get('current_round')}")
        else:
            print(f"  round {round_idx}: loss={loss:.4f}, upload FAILED: {result}")

    print(f"[{args.device_id}] done; uploaded {client.local_samples} samples total")


if __name__ == "__main__":
    main()
