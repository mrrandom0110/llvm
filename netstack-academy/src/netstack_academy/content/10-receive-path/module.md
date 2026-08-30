---
schema_version: 1
id: module-receive-path
slug: receive-path
title: The receive path
order: 1
summary: How a frame on the wire becomes an sk_buff a socket can read.
---

Receiving a packet is mostly a story about deferral. The device raises an
interrupt, the driver does as little as it possibly can, and the real work of
turning bytes into an `sk_buff` and walking it up through the protocol layers
happens later, in softirq context, under a budget.

This module follows that path from the device queue up to the socket.
