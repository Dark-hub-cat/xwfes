#pragma once

#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <cmath>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <pdh.h>
#include <pdhmsg.h>
#else
#include <cstdio>
#include <dirent.h>
#include <cstdlib>
#endif

namespace ft {

struct ThrottleConfig {
    bool enabled = true;
    int util_percent = 65;
    float temp_limit_c = 85.f;
    int window_ms = 200;
};

class DeviceGovernor {
public:
    explicit DeviceGovernor(ThrottleConfig cfg = {})
        : cfg_(cfg), eff_util_(cfg.util_percent) {
#ifdef _WIN32
        init_pdh();
#else
        scan_sysfs();
#endif
    }

    ~DeviceGovernor() {
#ifdef _WIN32
        if (query_) PdhCloseQuery(query_);
#endif
    }

    void before_submit() {
        if (!cfg_.enabled || eff_util_ >= 99.9f) return;
        const double t = now_ms();
        if (win_start_ < 0) win_start_ = t;
        thermal_tick(t);

        std::lock_guard<std::mutex> g(mu_);
        const double win = t - win_start_;
        const double allow = win * (eff_util_ / 100.0);
        if (busy_ms_ >= allow && win < (double)cfg_.window_ms) {
            const double sleep_ms =
                (double)cfg_.window_ms - win -
                (busy_ms_ - allow) * (double)cfg_.window_ms / std::max(1.0, allow);
            if (sleep_ms > 1.0) {
                auto dur = std::chrono::milliseconds((long)std::min(sleep_ms, 50.0));
                std::this_thread::sleep_for(dur);
                idle_ms_ += sleep_ms;
            }
        }
        submit_start_ = now_ms();
    }

    void after_submit() {
        const double t = now_ms();
        std::lock_guard<std::mutex> g(mu_);
        busy_ms_ += t - submit_start_;
        if (t - win_start_ >= (double)cfg_.window_ms) {
            win_start_ = t;
            busy_ms_ = 0;
            idle_ms_ = 0;
        }
    }

    float measured_util() {
#ifdef _WIN32
        return pdh_util();
#else
        return sysfs_util();
#endif
    }

    float temp_c() const { return last_temp_; }

    ThrottleConfig config() const { return cfg_; }
    void set_config(const ThrottleConfig& c) {
        cfg_ = c;
        eff_util_ = (float)c.util_percent;
    }

private:
    static double now_ms() {
        using namespace std::chrono;
        return duration<double, std::milli>(steady_clock::now().time_since_epoch())
            .count();
    }

    void thermal_tick(double t) {
        if (t - last_check_ < 2000.0) return;
        last_check_ = t;
#ifdef _WIN32
        last_temp_ = -1.f;
        float u = pdh_util();
#else
        last_temp_ = read_hwmon_temp();
        float u = sysfs_util_raw();
#endif
        if (u >= 0) last_meas_util_ = u;
        if (cfg_.temp_limit_c > 0 && last_temp_ >= cfg_.temp_limit_c) {
            eff_util_ = std::max(20.f, eff_util_ - 10.f);
        } else if (last_temp_ > 0 && last_temp_ < cfg_.temp_limit_c - 8.f &&
                   eff_util_ < (float)cfg_.util_percent) {
            eff_util_ = std::min((float)cfg_.util_percent, eff_util_ + 5.f);
        }
    }

#ifdef _WIN32
    void init_pdh() {
        if (PdhOpenQueryW(nullptr, 0, &query_) != ERROR_SUCCESS) return;
        PDH_STATUS s = PdhAddEnglishCounterW(
            query_, L"\\GPU Engine(*)\\Utilization Percentage", 0, &counter_);
        pdh_ok_ = (s == ERROR_SUCCESS);
    }

    float pdh_util() {
        if (!pdh_ok_) return -1.f;
        DWORD size1 = 0, size2 = 0, count = 0;
        PdhCollectQueryData(query_);
        Sleep(120);
        PdhCollectQueryData(query_);
        PDH_FMT_COUNTERVALUE_ITEM_W* items = nullptr;
        PdhGetFormattedCounterArrayW(counter_, PDH_FMT_DOUBLE, &size1, &count, nullptr);
        if (count == 0) return -1.f;
        items = (PDH_FMT_COUNTERVALUE_ITEM_W*)malloc(size1);
        if (!items) return -1.f;
        double total = 0;
        if (PdhGetFormattedCounterArrayW(counter_, PDH_FMT_DOUBLE, &size1, &count,
                                         items) == ERROR_SUCCESS) {
            for (DWORD i = 0; i < count; ++i)
                if (items[i].FmtValue.doubleValue > 0) total += items[i].FmtValue.doubleValue;
        }
        free(items);
        return (float)std::min(total, 100.0);
    }
    PDH_HQUERY query_ = nullptr;
    PDH_HCOUNTER counter_ = nullptr;
    bool pdh_ok_ = false;

#else
    void scan_sysfs() {
        DIR* d = opendir("/sys/class/drm");
        if (!d) return;
        while (dirent* e = readdir(d)) {
            std::string name = e->d_name;
            if (name.rfind("card", 0) != 0) continue;
            std::string base = "/sys/class/drm/" + name + "/device";
            std::string busy = base + "/gpu_busy_percent";
            if (FILE* f = fopen(busy.c_str(), "r")) {
                fclose(f);
                drm_base_ = base;
                break;
            }
        }
        closedir(d);
        if (!drm_base_.empty()) {
            std::string cmd = "ls " + drm_base_ + "/hwmon/*/temp1_input 2>/dev/null";
            FILE* p = popen(cmd.c_str(), "r");
            if (p) {
                char path[512] = {0};
                if (fgets(path, sizeof(path), p)) {
                    size_t l = strlen(path);
                    while (l && (path[l - 1] == '\n' || path[l - 1] == ' ')) path[--l] = 0;
                    hwmon_path_ = path;
                }
                pclose(p);
            }
        }
    }

    static bool read_int_file(const std::string& path, long& out) {
        FILE* f = fopen(path.c_str(), "r");
        if (!f) return false;
        long v = -1;
        if (fscanf(f, "%ld", &v) != 1) v = -1;
        fclose(f);
        if (v < 0) return false;
        out = v;
        return true;
    }

    float sysfs_util_raw() const {
        long v;
        if (!drm_base_.empty() && read_int_file(drm_base_ + "/gpu_busy_percent", v))
            return (float)v;
        return -1.f;
    }
    float sysfs_util() { return sysfs_util_raw(); }

    float read_hwmon_temp() {
        long v;
        if (!hwmon_path_.empty() && read_int_file(hwmon_path_, v)) return v / 1000.f;
        return -1.f;
    }
    std::string drm_base_, hwmon_path_;
#endif

    ThrottleConfig cfg_;
    float eff_util_ = 65.f;
    float last_temp_ = -1.f;
    float last_meas_util_ = -1.f;
    double win_start_ = -1;
    double submit_start_ = 0;
    double busy_ms_ = 0;
    double idle_ms_ = 0;
    double last_check_ = 0;
    std::mutex mu_;
};

} // namespace ft
