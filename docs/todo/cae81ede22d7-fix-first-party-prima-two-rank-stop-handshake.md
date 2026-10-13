---
id: cae81ede22d7
kind: bug
title: Fix first-party Prima two-rank STOP handshake
seq: 1
status: assigned
priority: p1
created: 2026-10-13T14:56:58.182848Z
weave: 1
assignee: codex-gpt6-sol
sprint: 436
sprint_id: 20ad6f7b-d3d3-5ebf-973c-0bbb999db96f
sprint_title: 'Sphere regression recovery: p2p sharding + local inference back on, bashy-built engines'
---

Hosted Bashy CI candidate -4 run 38416093564: all four native builds, ring transport, single-rank inference pass; two-rank rank1 timeout on all platforms. Source at reviewed f4ea9845: examples/main/main.cpp rank0 prints perf then calls llama_free_sockets; src/llama.cpp llama_free_sockets creates new nng PUSH, connects asynchronously, sends STOP with dontwait and ignores optional result, then waits ring reply; rank1 signal_thread waits recv. Investigate transport evidence and this lost-STOP hypothesis; make minimal first-party source fix preserving bounded two-rank shutdown, add focused regression coverage where possible. Commit with Sprint/Story/Story-ID trailers; no push/tag, no Dragon native build/container/full suite. Report if hypothesis disproven. Manager will review, push, tag new source version, update Bashy pin and hosted CI.

Delivery evidence: The producer's authenticated run review in `docs/plan-prima-two-rank-timeout.md` (Sprint 436) confirms both ranks warm up and rank 0 prints timings before rank 1 times out. The syscall trace was not retained, so the exact lost STOP in candidate -4 remains an inference. The source defect is definite: a fresh asynchronous dial can make an unchecked `dontwait` STOP send return empty. This change uses bounded, checked sends and receives, a STOP/ACK ring so senders stay alive through delivery confirmation, an atomic worker stop flag, and nonzero engine exit on handshake failure. `common/zmq_nng_shim_test.cpp` covers late and absent listeners. Dragon native tests were not run under the story restriction; hosted two-rank testing remains necessary.
