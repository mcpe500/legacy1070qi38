// Persistent thread pool without per-call allocation. parallel_for hands out chunks of [0, n) through an
// atomic counter, so uneven work (attention over different context lengths) balances itself.
// The calling thread takes part, so a pool of 4 uses 3 workers plus the caller (cores 0-3 under taskset).
#pragma once
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

class Pool {
public:
    struct FuncRef {
        void * obj;
        void (*call)(void *, int64_t, int64_t);
    };

    explicit Pool(int n_threads) : n_(n_threads < 1 ? 1 : n_threads) {
        worker_ids_.reserve(n_ - 1);
        for (int i = 1; i < n_; ++i) {
            workers_.emplace_back([this] { loop(); });
            worker_ids_.push_back(workers_.back().get_id());
        }
    }
    ~Pool() {
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_ = true;
            ++gen_;
        }
        cv_.notify_all();
        for (auto & t : workers_) t.join();
    }
    Pool(const Pool &) = delete;
    Pool & operator=(const Pool &) = delete;

    int size() const { return n_; }

    // f(begin, end) is called on disjoint chunks of at most `grain` items covering [0, n).
    template <class F> void parallel_for(int64_t n, int64_t grain, F && f) {
        using Fn = std::remove_reference_t<F>;
        FuncRef r{const_cast<void *>(static_cast<const void *>(&f)),
                  [](void * o, int64_t b, int64_t e) { (*static_cast<Fn *>(o))(b, e); }};
        run(n, grain < 1 ? 1 : grain, r);
    }

    // Stable work slot in [0, size()) for per-thread scratch carved from a static arena. The server calls
    // parallel_for from a different httplib thread for every request, so a TLS "first use" counter would hand the
    // same fallback slot to every later caller. The current caller is always slot 0; the persistent workers keep
    // slot 1..n-1. Used by the long-context attention scores, which no longer fit on the stack at 64K ctx.
    int slot() const {
        const std::thread::id me = std::this_thread::get_id();
        if (me == caller_.load(std::memory_order_acquire)) return 0;
        for (int i = 0; i < static_cast<int>(worker_ids_.size()); ++i)
            if (worker_ids_[i] == me) return i + 1;
        return 0;   // only reachable outside parallel_for
    }

private:
    void run(int64_t n, int64_t grain, FuncRef r) {
        caller_.store(std::this_thread::get_id(), std::memory_order_release);
        if (n <= 0) return;
        if (n_ == 1 || n <= grain) { r.call(r.obj, 0, n); return; }
        {
            std::lock_guard<std::mutex> lk(m_);
            fn_ = r;
            n_items_ = n;
            grain_ = grain;
            next_.store(0);
            busy_ = n_ - 1;
            ++gen_;
        }
        cv_.notify_all();
        drain();
        std::unique_lock<std::mutex> lk(m_);
        cv_done_.wait(lk, [this] { return busy_ == 0; });
    }

    void drain() {
        for (;;) {
            const int64_t b = next_.fetch_add(grain_);
            if (b >= n_items_) break;
            const int64_t e = b + grain_ < n_items_ ? b + grain_ : n_items_;
            fn_.call(fn_.obj, b, e);
        }
    }

    void loop() {
        uint64_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [&] { return gen_ != seen; });
                seen = gen_;
                if (stop_) return;
            }
            drain();
            {
                std::lock_guard<std::mutex> lk(m_);
                if (--busy_ == 0) cv_done_.notify_one();
            }
        }
    }

    int n_;
    std::vector<std::thread> workers_;
    std::mutex m_;
    std::condition_variable cv_, cv_done_;
    uint64_t gen_ = 0;
    int busy_ = 0;
    bool stop_ = false;
    FuncRef fn_{nullptr, nullptr};
    int64_t n_items_ = 0, grain_ = 1;
    std::atomic<int64_t> next_{0};
    std::atomic<std::thread::id> caller_{};
    std::vector<std::thread::id> worker_ids_;
};
