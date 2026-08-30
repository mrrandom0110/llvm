---
schema_version: 1
id: lesson-qdisc-dequeue
slug: qdisc-dequeue
title: Dequeueing from a qdisc
order: 10
status: published
summary: >-
  What sits between a socket's sk_buff and the driver's transmit routine, and
  which lock protects it.
objectives:
  - Name the two locks a software qdisc transmit path takes and what each guards
  - Explain why a qdisc may refuse to hand a packet to the driver
  - Describe what BQL bounds and why the byte count matters more than the packet count
prerequisites: []
packet_stage: tx-softirq
execution_context: >-
  Usually the process context of the sender, inside __dev_queue_xmit(), with
  preemption disabled. Deferred work runs in NET_TX_SOFTIRQ from the qdisc
  watchdog or when another CPU owns the qdisc.
ownership: >-
  A Qdisc belongs to a netdev transmit queue and is replaced as a unit under
  RTNL when the administrator changes it. An sk_buff handed to enqueue() is
  owned by the qdisc from that moment: the caller must not touch it again, and
  the qdisc is responsible for freeing it if it decides to drop it.
locking: >-
  Two separate locks. The qdisc's own spinlock (or its __QDISC_STATE_RUNNING
  seqcount for a lockless qdisc) serialises enqueue and dequeue against the
  queue structure; the netdev queue's _xmit_lock serialises the actual call
  into the driver's ndo_start_xmit. Holding the first while taking the second
  is the ordinary path, and never the reverse.
rcu: >-
  The qdisc a transmit takes is found under rcu_read_lock(), so a replacement
  installed under RTNL becomes visible to new transmits without stopping the
  ones already in flight.
structures:
  - name: struct Qdisc
    fields:
      - enqueue
      - dequeue
      - state
      - q
      - dev_queue
  - name: struct netdev_queue
    fields:
      - qdisc
      - _xmit_lock
      - dql
config_caveats:
  - >-
    CONFIG_BQL provides the dynamic queue limits that bound in-flight bytes per
    transmit queue; without it a driver's ring is bounded only by descriptors,
    which lets a fast NIC hold far more queued bytes than any AQM can manage.
  - >-
    CONFIG_NET_SCH_FQ_CODEL is the usual default qdisc; a kernel built without
    it falls back to pfifo_fast, which has no AQM at all.
version_caveats:
  - >-
    Lockless (NOLOCK) qdiscs changed the serialisation story substantially;
    read the Qdisc's TCQ_F_NOLOCK flag before assuming a spinlock is held.
tracepoints:
  - qdisc:qdisc_dequeue
  - qdisc:qdisc_enqueue
  - net:net_dev_queue
  - net:net_dev_xmit
source_symbols:
  - name: __dev_queue_xmit
    path: net/core/dev.c
  - name: sch_direct_xmit
    path: net/sched/sch_generic.c
  - name: qdisc_dequeue_head
lab:
  commands:
    - tc -s qdisc show
    - >-
      cat /sys/class/net/*/queues/tx-0/byte_queue_limits/limit
    - >-
      sudo timeout 5 perf stat -e 'qdisc:qdisc_dequeue' -a
  expected_observations:
    - >-
      tc -s reports backlog as both bytes and packets, plus a drop and an
      overlimit count. A non-zero overlimit with a small backlog is the shape
      of a rate limiter working; a large persistent backlog is the shape of
      bufferbloat.
    - >-
      The BQL limit adapts at runtime rather than staying at its initial value:
      reading it twice under load usually gives two different numbers.
  cleanup:
    - >-
      Nothing to undo: every command above only reads state.
quiz:
  - id: q-locks
    prompt: >-
      Which lock serialises the actual call into a driver's ndo_start_xmit?
    options:
      - id: a
        text: The Qdisc's own spinlock
      - id: b
        text: The netdev_queue's _xmit_lock
      - id: c
        text: RTNL
      - id: d
        text: The socket lock of the sending socket
    answer: b
    explanation: >-
      The two are distinct and both matter. The qdisc spinlock protects the
      queue structure during enqueue and dequeue; _xmit_lock protects the
      driver's transmit routine for that queue. RTNL is a configuration lock,
      far too heavy for a per-packet path, and the socket lock was released
      long before the packet reached the qdisc.
  - id: q-refuse
    prompt: >-
      A qdisc's dequeue() returns NULL even though its backlog is non-empty.
      What is the most likely reason?
    options:
      - id: a
        text: The queue structure is corrupt
      - id: b
        text: >-
          A shaper is holding the packet back until its scheduled transmit
          time, or the device queue is stopped
      - id: c
        text: The packet was dropped and the backlog counter is stale
      - id: d
        text: Another CPU is enqueueing, so dequeue must retry later
    answer: b
    explanation: >-
      dequeue() returning NULL with a backlog is the normal way a
      rate-limiting or time-based qdisc says "not yet": it arms the qdisc
      watchdog and returns nothing. The same happens when BQL or the driver
      has stopped the transmit queue, because handing a packet down then would
      overrun the ring.
  - id: q-bql
    prompt: What does byte queue limits (BQL) bound?
    options:
      - id: a
        text: The number of packets a qdisc may hold in its backlog
      - id: b
        text: The number of bytes handed to the driver's ring but not yet transmitted
      - id: c
        text: The rate in bytes per second at which a qdisc may dequeue
      - id: d
        text: The size in bytes of the largest sk_buff a driver will accept
    answer: b
    explanation: >-
      BQL bounds in-flight bytes below the qdisc, in the driver's ring. Bytes
      rather than packets is the point: a ring of 512 descriptors holds a very
      different amount of transmit latency depending on whether those are
      64-byte ACKs or 1500-byte segments, and only the byte count predicts how
      long a newly queued packet will wait.
mastery_gate:
  min_quiz_score: 0.67
  required_review_level: 2
---

## What is between the socket and the wire

`__dev_queue_xmit()` finds the transmit queue for the packet, finds that
queue's `Qdisc` under RCU, and -- if there is a real one -- hands the
`sk_buff` to `enqueue()`. From that moment the qdisc owns the packet. The
caller does not touch it again, and if the qdisc decides to drop it, the qdisc
frees it.

Then the same call tries to dequeue and transmit, because the common case is
that the queue was empty and the packet can go straight out:

```c
static inline int __dev_xmit_skb(struct sk_buff *skb, struct Qdisc *q,
                                 struct net_device *dev,
                                 struct netdev_queue *txq)
{
        spinlock_t *root_lock = qdisc_lock(q);
        int rc;

        spin_lock(root_lock);
        rc = q->enqueue(skb, q, &to_free) & NET_XMIT_MASK;
        __qdisc_run(q);
        spin_unlock(root_lock);

        return rc;
}
```

## Two locks, in one order

The qdisc spinlock and the queue's `_xmit_lock` guard different things, and
conflating them is the usual source of confusion when reading this path:

- the **qdisc lock** protects the queue *structure* -- enqueue, dequeue, the
  backlog counters, the shaper's state;
- the **`_xmit_lock`** protects the *driver* -- one CPU at a time inside
  `ndo_start_xmit` for a given transmit queue.

`sch_direct_xmit()` is where the second is taken, and it deliberately drops
the qdisc lock before doing so: the driver call can be slow, and holding the
structure lock across it would block every other CPU trying to enqueue.

## "Not yet" is a normal answer

`dequeue()` returning `NULL` does not mean the queue is empty. A shaper --
`tbf`, `htb`, `fq` -- returns `NULL` when the next packet is not due yet, and
arms the qdisc watchdog to come back in `NET_TX_SOFTIRQ`. A stopped transmit
queue produces the same answer for a different reason: the driver's ring is
full, or BQL has decided enough bytes are already in flight.

That last one is why BQL counts bytes. A ring of 512 descriptors is a very
different amount of queueing latency depending on whether it holds 64-byte
ACKs or 1500-byte segments, and it is the byte count -- not the packet count
-- that predicts how long a newly queued packet will actually wait.
