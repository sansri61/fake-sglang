"""Distance-dependent KV transfer.

The link a handoff crosses is decided by where the two workers sit, so the same
prompt must cost more to move between zones than within a node -- otherwise a
topology-aware router has nothing to win and a topology-blind one nothing to
lose.
"""

from __future__ import annotations

import json

import pytest

from sglang import Engine

from fakeengine.topology import DEFAULT_LINK, Topology, TopologyError
from test_disagg import PROMPT, _collect

TOPOLOGY = {
    "levels": ["zone", "rack", "node"],
    "links": {
        "node": {"bandwidth_gb_s": 400, "latency_ms": 0.01},
        "rack": {"bandwidth_gb_s": 50, "latency_ms": 0.05},
        "zone": {"bandwidth_gb_s": 10, "latency_ms": 0.5},
        "default": {"bandwidth_gb_s": 1, "latency_ms": 5},
    },
}


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ("z1/r1/n1", "z1/r1/n1", "node"),
        ("z1/r1/n1", "z1/r1/n2", "rack"),
        ("z1/r1/n1", "z1/r2/n1", "zone"),
        ("z1/r1/n1", "z2/r1/n1", DEFAULT_LINK),
    ],
)
def test_link_is_the_deepest_shared_level(a, b, expected):
    topo = Topology.from_dict(TOPOLOGY)
    assert topo.link_between(a, b).name == expected
    assert topo.link_between(b, a).name == expected


def test_same_names_in_different_parents_are_not_the_same_place():
    """rack-1 in zone A and rack-1 in zone B share nothing."""
    topo = Topology.from_dict(TOPOLOGY)
    assert topo.link_between("a/rack-1/n1", "b/rack-1/n1").name == DEFAULT_LINK


@pytest.mark.parametrize(
    "mutate",
    [
        lambda t: t["links"].pop("rack"),
        lambda t: t["links"].update(extra={"bandwidth_gb_s": 1}),
        lambda t: t["links"]["node"].update(bandwidth_gb_s=0),
        lambda t: t.update(levels=[]),
        lambda t: t.update(levels=["zone", "zone", "node"]),
    ],
)
def test_malformed_topology_is_rejected(mutate):
    raw = json.loads(json.dumps(TOPOLOGY))
    mutate(raw)
    with pytest.raises(TopologyError):
        Topology.from_dict(raw)


def test_location_depth_must_match_levels():
    topo = Topology.from_dict(TOPOLOGY)
    with pytest.raises(TopologyError):
        topo.link_between("z1/r1", "z1/r1/n1")


def test_worker_with_bad_location_fails_at_startup(server_args):
    with pytest.raises(TopologyError):
        Engine(
            server_args=server_args(
                disaggregation_mode="decode",
                fake_location="z1/r1",
                fake_topology=json.dumps(TOPOLOGY),
            )
        )


async def _transfer(prefill, decode, room, port):
    stream = await prefill.async_generate(
        input_ids=PROMPT,
        sampling_params={"max_new_tokens": 1},
        stream=True,
        bootstrap_host="127.0.0.1",
        bootstrap_port=port,
        bootstrap_room=room,
    )
    await _collect(stream)

    from fakeengine.transfer import KVReceiver

    receiver = KVReceiver(
        decode.sim_config,
        decode._bootstrap_client,
        bootstrap_host="127.0.0.1",
        bootstrap_port=port,
        bootstrap_room=room,
    )
    return await receiver.receive(len(PROMPT))


async def test_transfer_cost_grows_with_distance(server_args, unused_port):
    topo = json.dumps(TOPOLOGY)
    placed = dict(fake_topology=topo)
    prefill = Engine(
        server_args=server_args(
            disaggregation_mode="prefill",
            disaggregation_bootstrap_port=unused_port,
            fake_location="z1/r1/n1",
            **placed,
        )
    )
    decodes = {
        name: Engine(
            server_args=server_args(
                disaggregation_mode="decode", fake_location=loc, **placed
            )
        )
        for name, loc in [
            ("node", "z1/r1/n1"),
            ("rack", "z1/r1/n2"),
            ("zone", "z1/r2/n1"),
            (DEFAULT_LINK, "z2/r1/n1"),
        ]
    }
    try:
        results = {}
        for room, (name, decode) in enumerate(decodes.items(), start=2001):
            results[name] = await _transfer(prefill, decode, room, unused_port)

        assert {name: r["link"] for name, r in results.items()} == {
            name: name for name in results
        }
        costs = [results[n]["transfer_ms"] for n in ("node", "rack", "zone", DEFAULT_LINK)]
        assert costs == sorted(costs) and len(set(costs)) == 4
    finally:
        await prefill._async_shutdown()
        for decode in decodes.values():
            await decode._async_shutdown()


async def test_unplaced_workers_keep_the_global_link(server_args, unused_port):
    """A topology with no location on one end falls back, rather than guessing."""
    prefill = Engine(
        server_args=server_args(
            disaggregation_mode="prefill",
            disaggregation_bootstrap_port=unused_port,
            fake_topology=json.dumps(TOPOLOGY),
        )
    )
    decode = Engine(
        server_args=server_args(
            disaggregation_mode="decode",
            fake_location="z1/r1/n1",
            fake_topology=json.dumps(TOPOLOGY),
        )
    )
    try:
        metadata = await _transfer(prefill, decode, 2101, unused_port)
        assert metadata["link"] is None
    finally:
        await prefill._async_shutdown()
        await decode._async_shutdown()
