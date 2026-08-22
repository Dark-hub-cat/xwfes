#include "ft/engine.h"
#include "ft/spec.h"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

using namespace ft;

namespace {

struct Args {
    std::string model;
    std::string draft_model;
    std::string spec = "none";
    int spec_k = 4;
    float p_min = 0.05f;
    float p_split = 0.9f;
    int dflash_rounds = 3;
    int threads = 0;
    std::string igpu = "auto";
    int64_t chunk_neurons = 0;
    int staging_mb = 16;
    int64_t ctx = 0;
    float temp = 0.7f;
    int top_k = 40;
    float top_p = 0.95f;
    uint64_t seed = 42;
    std::string prompt;
    int max_new = 256;
    int gpu_util = 65;
    float gpu_temp_limit = 85.f;
    bool no_governor = false;
};

void usage() {
    std::printf(
        "ft-cli - FreeToken iGPU prototype\n\n"
        "  --model PATH          target GGUF (required)\n"
        "  --draft-model PATH    small GGUF for speculative drafting\n"
        "  --spec MODE           none|ngram|draft|dflash|mlsd (default none)\n"
        "  --spec-k N            max proposals per cycle (default 4)\n"
        "  --dflash-rounds N     DFlash-style refinement rounds (default 3)\n"
        "  --threads N           CPU threads (default: hardware)\n"
        "  --igpu MODE           auto|on|off (default auto)\n"
        "  --chunk-neurons N     expert fragment width (default: auto)\n"
        "  --staging-mb N        expert staging budget MB (default 16)\n"
        "  --ctx N               context length override\n"
        "  --temp F --top-k N --top-p F --seed N\n"
        "  --prompt TEXT         run once and exit (otherwise REPL)\n"
        "  -n, --max-new N       max new tokens (default 256)\n"
        "  --gpu-util P          accelerator duty-cycle percent cap (default 65)\n"
        "  --gpu-temp-limit C    thermal backoff threshold (default 85)\n"
        "  --no-governor         disable GPU/iGPU governor\n");
}

IgpuMode parse_igpu(const std::string& s) {
    if (s == "on") return IgpuMode::On;
    if (s == "off") return IgpuMode::Off;
    return IgpuMode::Auto;
}

std::unique_ptr<IDrafter> make_drafter(const Args& a, const EngineOptions& base) {
    if (a.spec == "none") return nullptr;
    if (a.spec == "ngram") return std::make_unique<NgramDrafter>(4);
    EngineOptions dopt = base;
    dopt.model_path = a.draft_model;
    dopt.ctx_len = base.ctx_len ? base.ctx_len : 2048;
    if (a.spec == "draft") return std::make_unique<DraftModelDrafter>(dopt);
    if (a.spec == "dflash")
        return std::make_unique<DFlashBlockDrafter>(dopt, a.dflash_rounds);
    if (a.spec == "mlsd") {
        auto multi = std::make_unique<MultiDrafter>();
        multi->add(std::make_unique<NgramDrafter>(4));
        if (!a.draft_model.empty())
            multi->add(std::make_unique<DFlashBlockDrafter>(dopt, a.dflash_rounds));
        return multi;
    }
    throw std::runtime_error("unknown --spec mode: " + a.spec);
}

void print_stats(const RunStats& st) {
    std::printf("[stats] gen %.2f t/s (%llu tok), prefill %.1f t/s | "
                "spec %llu/%llu accepted (%llu cycles)\n",
                st.gen_tps(), (unsigned long long)st.gen_tokens,
                st.prefill_tps(),
                (unsigned long long)st.spec_accepted,
                (unsigned long long)st.spec_proposed,
                (unsigned long long)st.spec_cycles);
}

void run_prompt(Engine& eng, IDrafter* dr, const Args& a,
                const SampleParams& sp) {
    Tokenizer tk = eng.tok();
    auto ids = tk.encode(a.prompt, tk.bos() >= 0);
    eng.prefill(ids);
    GenResult g = generate(eng, dr, sp, a.max_new, a.spec_k,
                           [](token_t, const std::string& piece) {
                               std::fputs(piece.c_str(), stdout);
                               std::fflush(stdout);
                           });
    std::printf("\n");
    print_stats(eng.stats());
}

} // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string f = argv[i];
        auto next = [&](const char* def = "") -> std::string {
            if (i + 1 >= argc) return def;
            return argv[++i];
        };
        if (f == "--model") a.model = next();
        else if (f == "--draft-model") a.draft_model = next();
        else if (f == "--spec") a.spec = next();
        else if (f == "--spec-k") a.spec_k = std::atoi(next().c_str());
        else if (f == "--p-min") a.p_min = (float)std::atof(next().c_str());
        else if (f == "--p-split") a.p_split = (float)std::atof(next().c_str());
        else if (f == "--dflash-rounds") a.dflash_rounds = std::atoi(next().c_str());
        else if (f == "--threads") a.threads = std::atoi(next().c_str());
        else if (f == "--igpu") a.igpu = next();
        else if (f == "--chunk-neurons") a.chunk_neurons = std::atoll(next().c_str());
        else if (f == "--staging-mb") a.staging_mb = std::atoi(next().c_str());
        else if (f == "--ctx") a.ctx = std::atoll(next().c_str());
        else if (f == "--temp") a.temp = (float)std::atof(next().c_str());
        else if (f == "--top-k") a.top_k = std::atoi(next().c_str());
        else if (f == "--top-p") a.top_p = (float)std::atof(next().c_str());
        else if (f == "--seed") a.seed = std::strtoull(next().c_str(), nullptr, 10);
        else if (f == "--prompt") a.prompt = next();
        else if (f == "-n" || f == "--max-new") a.max_new = std::atoi(next().c_str());
        else if (f == "--gpu-util") a.gpu_util = std::atoi(next().c_str());
        else if (f == "--gpu-temp-limit") a.gpu_temp_limit = (float)std::atof(next().c_str());
        else if (f == "--no-governor") a.no_governor = true;
        else if (f == "-h" || f == "--help") { usage(); return 0; }
        else { std::fprintf(stderr, "unknown flag: %s\n", f.c_str()); usage(); return 2; }
    }
    if (a.model.empty()) {
        std::fprintf(stderr, "--model is required\n");
        usage();
        return 2;
    }
    if ((a.spec == "draft" || a.spec == "dflash" || a.spec == "mlsd") &&
        a.draft_model.empty()) {
        std::fprintf(stderr, "--spec %s requires --draft-model\n", a.spec.c_str());
        return 2;
    }

    try {
        EngineOptions eo;
        eo.model_path = a.model;
        eo.threads = a.threads;
        eo.igpu = parse_igpu(a.igpu);
        eo.chunk_neurons = a.chunk_neurons;
        eo.staging_budget = (int64_t)a.staging_mb << 20;
        eo.ctx_len = a.ctx;
        eo.throttle.enabled = !a.no_governor;
        eo.throttle.util_percent = a.gpu_util;
        eo.throttle.temp_limit_c = a.gpu_temp_limit;

        Engine eng;
        eng.load(eo);
        std::printf("[plan] %s\n", eng.cfg().describe().c_str());
        std::printf("%s\n", eng.sched().describe().c_str());

        SampleParams sp;
        sp.temp = a.temp;
        sp.top_k = a.top_k;
        sp.top_p = a.top_p;
        sp.seed = a.seed;

        auto drafter = make_drafter(a, eo);

        if (!a.prompt.empty()) {
            run_prompt(eng, drafter.get(), a, sp);
            return 0;
        }

        std::printf("[repl] enter a prompt; commands: /stats /reset /quit\n");
        std::string line;
        while (true) {
            std::printf("\n> ");
            std::fflush(stdout);
            if (!std::getline(std::cin, line)) break;
            if (line == "/quit" || line == "/exit") break;
            if (line == "/reset") { eng.reset(); continue; }
            if (line == "/stats") { print_stats(eng.stats()); continue; }
            if (line.rfind("/igpu ", 0) == 0) {
                std::string m = line.substr(6);
                try {
                    eng.sched().set_igpu_enabled(m == "on");
                    std::printf("[plan] iGPU %s\n", m == "on" ? "enabled" : "disabled");
                } catch (const std::exception& e) {
                    std::printf("[err] %s\n", e.what());
                }
                continue;
            }
            if (line.empty()) continue;
            Args one = a;
            one.prompt = line;
            run_prompt(eng, drafter.get(), one, sp);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[fatal] %s\n", e.what());
        return 1;
    }
    return 0;
}
