---
schema_version: 1
id: module-transmit-path
slug: transmit-path
title: The transmit path
order: 2
summary: How an sk_buff written by a socket reaches the device queue.
---

Transmission is a story about queueing. Between the socket that produced a
packet and the device that will put it on the wire sits a queueing discipline,
which decides the order packets leave in and whether they leave at all.

This module follows that path from the qdisc down to the driver.
