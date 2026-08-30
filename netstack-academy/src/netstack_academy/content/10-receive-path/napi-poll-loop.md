---
schema_version: 1
id: lesson-napi-poll-loop
slug: napi-poll-loop
title: The NAPI poll loop
order: 10
status: published
summary: >-
  Why the receive path runs in softirq context under a budget, and what that
  budget actually bounds.
objectives:
  - Explain why a driver's hard IRQ handler does almost no work
  - Name the budget that bounds one NAPI poll and say what it counts
  - Describe what happens when a poll exhausts its budget
prerequisites: []
packet_stage: rx-softirq
execution_context: >-
  NET_RX_SOFTIRQ, on the CPU that took the device interrupt unless RPS moved
  the work elsewhere. Hard interrupts are enabled; preemption is disabled.
ownership: >-
  A napi_struct belongs to the device driver that registered it, usually one
  per receive queue. The stack borrows it for the length of a poll and never
  outlives the driver's netif_napi_del().
locking: >-
  NAPI_STATE_SCHED is the serialisation: exactly one CPU can own a poll for a
  given napi_struct at a time, and it is set before the instance is placed on
  a softnet_data poll_list. There is no lock around the driver's poll method
  itself, which is why a driver may touch its own rings without one.
rcu: >-
  rcu_read_lock() is held across delivery to the protocol handlers, so a
  packet_type or an rx_handler can be removed concurrently without the receive
  path taking a reference on each packet.
structures:
  - name: struct napi_struct
    fields:
      - poll
      - weight
      - state
      - poll_list
  - name: struct softnet_data
    fields:
      - poll_list
      - process_queue
      - time_squeeze
config_caveats:
  - >-
    CONFIG_RPS lets the receive path enqueue to a remote CPU's backlog instead
    of continuing on the interrupting CPU, which moves the protocol work but
    not the poll itself.
  - >-
    CONFIG_NET_RX_BUSY_POLL lets a socket drive the device's poll method
    directly from a read, bypassing the softirq entirely for latency.
version_caveats:
  - >-
    The per-poll budget and the overall netdev_budget/netdev_budget_usecs pair
    have been tuned repeatedly; treat the numbers as defaults to read from
    /proc/sys/net/core, not as constants.
tracepoints:
  - napi:napi_poll
  - net:netif_receive_skb
  - net:netif_rx
source_symbols:
  - name: napi_poll
    path: net/core/dev.c
  - name: net_rx_action
    path: net/core/dev.c
  - name: netif_receive_skb
lab:
  commands:
    - cat /proc/net/softnet_stat
    - sysctl net.core.netdev_budget net.core.netdev_budget_usecs
    - >-
      sudo timeout 5 perf stat -e 'napi:napi_poll' -a
  expected_observations:
    - >-
      Each row of softnet_stat is one CPU. The third column is time_squeeze:
      the number of times net_rx_action ran out of budget with work still
      pending, which is the counter that tells you the budget is binding.
    - >-
      netdev_budget bounds packets across all instances in one softirq run,
      while each napi_struct's own weight bounds one instance's poll; they are
      different limits and both apply.
  cleanup:
    - >-
      Nothing to undo: every command above only reads counters.
quiz:
  - id: q-context
    prompt: In which context does a driver's NAPI poll method normally run?
    options:
      - id: a
        text: The device's hard IRQ handler
      - id: b
        text: NET_RX_SOFTIRQ, after the hard IRQ handler has scheduled it
      - id: c
        text: A dedicated kernel thread per device, always
      - id: d
        text: Process context, on the CPU that calls recvmsg()
    answer: b
    explanation: >-
      The hard IRQ handler does almost nothing: it masks device interrupts and
      calls napi_schedule(), which sets NAPI_STATE_SCHED and raises
      NET_RX_SOFTIRQ. The poll itself runs from net_rx_action in softirq
      context. Threaded NAPI exists but is opt-in, and busy polling is driven
      from a socket read rather than being the normal path.
  - id: q-budget
    prompt: What does a napi_struct's weight bound?
    options:
      - id: a
        text: How many bytes one poll may copy
      - id: b
        text: How many packets one call to that instance's poll method may process
      - id: c
        text: How long in microseconds one softirq run may last
      - id: d
        text: How many CPUs may poll the instance concurrently
    answer: b
    explanation: >-
      weight is a packet count for one poll of one instance. The time bound is
      a separate sysctl (netdev_budget_usecs), and concurrency is not a budget
      at all -- NAPI_STATE_SCHED already guarantees a single poller.
  - id: q-squeeze
    prompt: >-
      A poll returns having used its entire budget and the device still has
      packets queued. What happens?
    options:
      - id: a
        text: The remaining packets are dropped and a counter is incremented
      - id: b
        text: >-
          The instance stays on the poll list and is polled again, rather than
          re-enabling device interrupts
      - id: c
        text: The driver re-enables interrupts and waits for the next one
      - id: d
        text: The softirq loops on that instance until the queue drains
    answer: b
    explanation: >-
      Exhausting the budget means there is more work, so the driver does not
      call napi_complete_done() and the instance is left on the poll list.
      net_rx_action will come back to it, and if the overall budget is gone
      too it increments time_squeeze and defers the rest to the next softirq
      -- work is postponed, never dropped, at this layer.
mastery_gate:
  min_quiz_score: 0.67
  required_review_level: 2
---

## Why the hard IRQ handler does nothing

A device that has received a frame raises an interrupt. Servicing it in the
handler would be the obvious thing to do and close to the worst thing to do:
hard IRQ context runs with interrupts disabled on that CPU, cannot sleep, and
holds up everything else on the machine. At line rate on a modern NIC the
interrupts would also arrive faster than the handler could retire them, and
the machine would spend all its time entering and leaving interrupt context
instead of processing packets -- a receive livelock.

So the handler does two things and returns: it tells the device to stop
raising receive interrupts, and it calls `napi_schedule()`.

## What napi_schedule actually schedules

`napi_schedule()` sets the `NAPI_STATE_SCHED` bit on the instance, puts it on
the current CPU's `softnet_data.poll_list`, and raises `NET_RX_SOFTIRQ`. That
bit is the whole concurrency story for NAPI: it is set with an atomic
test-and-set, so a second CPU that tries to schedule the same instance loses
and does nothing. Exactly one CPU polls a given `napi_struct` at a time, which
is why a driver's poll method may walk its own receive rings without taking
any lock.

`net_rx_action` is what runs when the softirq fires. It walks the poll list
and calls each instance's `poll` method with a budget:

```c
static int __napi_poll(struct napi_struct *n, bool *repoll)
{
        int weight = n->weight;
        int work;

        work = n->poll(n, weight);

        if (likely(work < weight))
                return work;
        /* Drained fewer than allowed: the queue is empty. */

        *repoll = true;
        return work;
}
```

## Two budgets, not one

There are two limits, and confusing them makes `/proc/net/softnet_stat`
unreadable:

| Limit | Scope | Counts |
| --- | --- | --- |
| `napi_struct.weight` | one poll of one instance | packets |
| `net.core.netdev_budget` | one `net_rx_action` run, all instances | packets |
| `net.core.netdev_budget_usecs` | one `net_rx_action` run | time |

A driver signals "I am done, re-enable my interrupts" by processing *fewer*
packets than its weight and calling `napi_complete_done()`. Returning the full
weight means the opposite: there is more work, leave me on the list. When
`net_rx_action` runs out of its own overall budget with instances still on the
list, it increments `time_squeeze` for that CPU and returns, leaving the rest
for the next softirq.

That counter is the one worth watching. A steadily climbing `time_squeeze`
means the receive path is being cut off mid-drain, and it is the evidence you
want before touching `netdev_budget`.

## Where the packet goes

Whatever the driver's poll method builds, it hands upward through
`napi_gro_receive()` or `netif_receive_skb()`. From there the packet is
delivered under `rcu_read_lock()` to any `rx_handler` the device has, then to
the `packet_type` registered for its protocol -- `ip_rcv()` for IPv4. The RCU
read side is what lets a protocol handler be unregistered without the receive
path paying for a reference count on every single packet.
