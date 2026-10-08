"""launch/cluster.py turns a YAML file into the right processes and flags."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("cluster", ROOT / "launch" / "cluster.py")
cluster = importlib.util.module_from_spec(_spec)
sys.modules["cluster"] = cluster    # dataclasses resolve their module by name
_spec.loader.exec_module(cluster)

TOPOLOGY = {
    "levels": ["rack", "node"],
    "links": {
        "node": {"bandwidth_gb_s": 400},
        "rack": {"bandwidth_gb_s": 50},
        "default": {"bandwidth_gb_s": 10},
    },
}


def _build(config, extra=(), offset=0):
    return cluster.build(config, list(extra), "python", offset)


def _flag(proc, name):
    return proc.argv[proc.argv.index(name) + 1] if name in proc.argv else None


@pytest.mark.parametrize("path", sorted((ROOT / "launch" / "clusters").glob("*.yaml")))
def test_shipped_examples_are_valid(path):
    procs, _ = _build(yaml.safe_load(path.read_text()))
    assert procs[0].name == "frontend"


def test_counts_roles_and_unique_ports():
    procs, _ = _build(
        {
            "frontend": {"router_mode": "kv"},
            "topology": TOPOLOGY,
            "workers": [
                {"role": "prefill", "count": 2, "location": "r1/n1"},
                {"role": "decode", "count": 3, "location": "r2/n1"},
                {"role": "agg", "location": "r1/n2"},
            ],
        }
    )
    workers = procs[1:]
    assert [p.name for p in workers] == [
        "prefill-0", "prefill-1", "decode-0", "decode-1", "decode-2", "agg-0",
    ]
    assert [_flag(p, "--disaggregation-mode") for p in workers] == (
        ["prefill"] * 2 + ["decode"] * 3 + [None]
    )
    ports = [port for p in procs for port in p.ports.values()]
    assert len(ports) == len(set(ports))
    # Every worker publishes KV events under the kv router, each on its own port.
    assert all(_flag(p, "--kv-events-config") for p in workers)
    # And every worker carries the same topology table.
    assert len({_flag(p, "--fake-topology") for p in workers}) == 1
    assert json.loads(_flag(workers[0], "--fake-topology"))["levels"] == ["rack", "node"]


def test_port_offset_and_arg_layering():
    procs, _ = _build(
        {
            "defaults": {"args": ["--fake-itl-ms", "5"]},
            "workers": [{"role": "agg", "args": "--fake-speedup-ratio 2"}],
        },
        extra=["--fake-num-kv-blocks", "64"],
        offset=100,
    )
    frontend, worker = procs
    assert frontend.ports["http"] == 8100
    assert worker.env["DYN_SYSTEM_PORT"] == "8181"
    tail = worker.argv[-6:]
    assert tail == ["--fake-itl-ms", "5", "--fake-speedup-ratio", "2", "--fake-num-kv-blocks", "64"]
    assert _flag(worker, "--kv-events-config") is None    # no kv router, no events


@pytest.mark.parametrize(
    "config, message",
    [
        ({"workers": [{"role": "prefill"}]}, "come as a pair"),
        ({"workers": [{"role": "gpu"}]}, "role must be one of"),
        ({"workers": [{"role": "agg", "location": "r1/n1"}]}, "no topology"),
        ({"topology": TOPOLOGY, "workers": [{"role": "agg"}]}, "needs a location"),
        ({"topology": TOPOLOGY, "workers": [{"role": "agg", "location": "r1"}]}, "components"),
        ({"workers": []}, "at least one"),
    ],
)
def test_bad_configs_fail_before_anything_starts(config, message):
    with pytest.raises(cluster.ConfigError, match=message):
        _build(config)
