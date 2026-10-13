// Standalone round-trip test for zmq_nng_shim.hpp.
//
// Build (static nng, after `make` has fetched+built deps/nng):
//   clang++ -std=c++17 -Ideps/nng/include \
//       common/zmq_nng_shim_test.cpp deps/nng/build/libnng.a \
//       -o ./zmq_nng_shim_test
//
// Exercises the exact API surface prima.cpp depends on, in the same
// bind/connect orientation prima uses (PULL binds, PUSH connects):
//   1. 3-part multipart round-trip (empty part + small part + ~1MB part)
//      asserted byte-for-byte.
//   2. rcvtimeo: a recv with no sender returns the empty optional within
//      ~500ms instead of hanging.
//   3. Shutdown STOP waits for a late pipe and times out if none appears.
//   4. Two ranks complete STOP and ACK before closing their senders.

#include "zmq_nng_shim.hpp"

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

static bool bytes_equal(const zmq::message_t & m, const std::vector<uint8_t> & v) {
    if (m.size() != v.size()) return false;
    if (v.empty()) return true;
    return std::memcmp(m.data(), v.data(), v.size()) == 0;
}

int main() {
    const std::string url = "tcp://127.0.0.1:5599";

    // ---- Test 1: multipart round-trip ------------------------------------
    {
        zmq::context_t ctx(2);
        zmq::socket_t puller(ctx, zmq::socket_type::pull);
        zmq::socket_t pusher(ctx, zmq::socket_type::push);

        puller.bind(url);                 // PULL binds  (matches prima recv_socket)
        pusher.connect(url);              // PUSH connects (matches prima send_socket)

        // give the dial/listen pipe a moment to establish
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Build 3 parts: empty, a small string, and a ~1MB pattern buffer.
        std::vector<uint8_t> p0;                                  // empty part
        std::vector<uint8_t> p1 = {'h','e','l','l','o','-',0x00,0x01,0x02,0xff};
        std::vector<uint8_t> p2(1u << 20);                        // 1 MiB
        for (size_t i = 0; i < p2.size(); ++i) p2[i] = (uint8_t)((i * 2654435761u) >> 13);

        std::vector<zmq::message_t> send_msgs;
        send_msgs.emplace_back(p0.data(), p0.size());
        send_msgs.emplace_back(p1.data(), p1.size());
        send_msgs.emplace_back(p2.data(), p2.size());

        zmq::send_multipart(pusher, send_msgs);

        std::vector<zmq::message_t> recv_msgs;
        auto n = zmq::recv_multipart(puller, std::back_inserter(recv_msgs));

        assert(n.has_value() && "recv_multipart returned timeout");
        assert(*n == 3 && "expected 3 parts");
        assert(recv_msgs.size() == 3);
        assert(bytes_equal(recv_msgs[0], p0) && "part0 (empty) mismatch");
        assert(bytes_equal(recv_msgs[1], p1) && "part1 (small) mismatch");
        assert(bytes_equal(recv_msgs[2], p2) && "part2 (1MB) mismatch");
        assert(recv_msgs[0].size() == 0);
        assert(recv_msgs[2].size() == (1u << 20));

        std::printf("  [1] multipart 3-part round-trip (0B + 10B + 1MiB): OK\n");
    }

    // ---- Test 2: rcvtimeo on idle socket ---------------------------------
    {
        zmq::context_t ctx(2);
        zmq::socket_t puller(ctx, zmq::socket_type::pull);
        puller.bind("tcp://127.0.0.1:5600");      // no sender ever connects
        puller.set(zmq::sockopt::rcvtimeo, 500);

        auto t0 = std::chrono::steady_clock::now();
        std::vector<zmq::message_t> msgs;
        auto r = zmq::recv_multipart(puller, std::back_inserter(msgs));
        auto t1 = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

        assert(!r.has_value() && "expected timeout (empty optional)");
        assert(ms >= 400 && ms < 1500 && "timeout fired outside ~500ms window");
        std::printf("  [2] rcvtimeo=500ms returned timeout in %lldms (no hang): OK\n",
                    (long long)ms);
    }

    // ---- Test 3: single-message send/recv + to_string --------------------
    {
        zmq::context_t ctx(2);
        zmq::socket_t puller(ctx, zmq::socket_type::pull);
        zmq::socket_t pusher(ctx, zmq::socket_type::push);
        puller.bind("tcp://127.0.0.1:5601");
        pusher.connect("tcp://127.0.0.1:5601");
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        zmq::message_t out("STOP", 4);
        pusher.send(out, zmq::send_flags::dontwait);

        zmq::message_t in;
        auto rr = puller.recv(in, zmq::recv_flags::none);
        assert(rr.has_value() && *rr == 4);
        assert(in.to_string() == "STOP");
        std::printf("  [3] single send/recv message_t(\"STOP\",4) + to_string: OK\n");
    }

    // ---- Test 4: fresh asynchronous dial cannot use a one-shot send -----
    {
        zmq::context_t ctx(2);
        zmq::socket_t pusher(ctx, zmq::socket_type::push);
        pusher.connect("tcp://127.0.0.1:5602");
        auto dropped = pusher.send(zmq::message_t("STOP", 4), zmq::send_flags::dontwait);
        assert(!dropped && "fresh dial unexpectedly had a pipe");

        pusher.set(zmq::sockopt::sndtimeo, 3000);
        zmq::socket_t puller(ctx, zmq::socket_type::pull);
        std::thread late_listener([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            puller.bind("tcp://127.0.0.1:5602");
        });
        auto sent = pusher.send(zmq::message_t("STOP", 4), zmq::send_flags::none);
        late_listener.join();
        assert(sent && *sent == 4 && "bounded send did not wait for the late pipe");
        puller.set(zmq::sockopt::rcvtimeo, 3000);
        zmq::message_t received;
        auto got = puller.recv(received, zmq::recv_flags::none);
        assert(got && *got == 4 && received.to_string() == "STOP");
        std::printf("  [4] STOP reaches a listener bound after connect: OK\n");
    }

    // ---- Test 5: shutdown send stays bounded when no rank listens -------
    {
        zmq::context_t ctx(2);
        zmq::socket_t pusher(ctx, zmq::socket_type::push);
        pusher.set(zmq::sockopt::sndtimeo, 300);
        pusher.connect("tcp://127.0.0.1:5603");
        auto t0 = std::chrono::steady_clock::now();
        auto sent = pusher.send(zmq::message_t("STOP", 4), zmq::send_flags::none);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count();
        assert(!sent && "send without a listener should time out");
        assert(ms >= 200 && ms < 1500 && "send timeout exceeded shutdown bound");
        std::printf("  [5] STOP send without a listener timed out in %lldms: OK\n", (long long) ms);
    }

    // ---- Test 6: two-rank STOP/ACK ring -------------------------------
    {
        zmq::context_t ctx(2);
        zmq::socket_t rank0_in(ctx, zmq::socket_type::pull);
        zmq::socket_t rank1_in(ctx, zmq::socket_type::pull);
        rank0_in.bind("tcp://127.0.0.1:5604");
        rank1_in.bind("tcp://127.0.0.1:5605");
        rank0_in.set(zmq::sockopt::rcvtimeo, 3000);
        rank1_in.set(zmq::sockopt::rcvtimeo, 3000);

        std::thread worker([&] {
            zmq::socket_t sender(ctx, zmq::socket_type::push);
            sender.set(zmq::sockopt::sndtimeo, 3000);
            sender.connect("tcp://127.0.0.1:5604");
            zmq::message_t in;
            assert(rank1_in.recv(in, zmq::recv_flags::none) && in.to_string() == "STOP");
            assert(sender.send(in, zmq::send_flags::none));
            assert(rank1_in.recv(in, zmq::recv_flags::none) && in.to_string() == "ACK");
            assert(sender.send(in, zmq::send_flags::none));
        });

        zmq::socket_t sender(ctx, zmq::socket_type::push);
        sender.set(zmq::sockopt::sndtimeo, 3000);
        sender.connect("tcp://127.0.0.1:5605");
        assert(sender.send(zmq::message_t("STOP", 4), zmq::send_flags::none));
        zmq::message_t in;
        assert(rank0_in.recv(in, zmq::recv_flags::none) && in.to_string() == "STOP");
        assert(sender.send(zmq::message_t("ACK", 3), zmq::send_flags::none));
        assert(rank0_in.recv(in, zmq::recv_flags::none) && in.to_string() == "ACK");
        worker.join();
        std::printf("  [6] two-rank STOP/ACK round trip: OK\n");
    }

    std::printf("PASS\n");
    return 0;
}
