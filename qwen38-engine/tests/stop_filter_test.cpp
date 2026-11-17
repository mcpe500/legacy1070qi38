// Unit test of src/engine/stop_filter.hpp: for every sample, cutting the generated text into pieces at every position (1, 2
// and 3 pieces, and one byte at a time) must give exactly the same sent text and the same hit flag as one piece, and the
// sent text must equal the reference: the text before the first stop string (after the reasoning block when thinking is on).
#include <cstdio>
#include <string>
#include <vector>

#include "stop_filter.hpp"

namespace {

struct Out { std::string text; bool hit; };

Out run(const std::vector<std::string> & stops, bool think, const std::vector<std::string> & pieces) {
    StopFilter f(stops, think);
    Out o{"", false};
    for (const std::string & p : pieces) {
        o.text += f.push(p);
        if (f.hit()) break;
    }
    if (!f.hit()) o.text += f.flush();
    o.hit = f.hit();
    return o;
}

// Reference: first occurrence of any stop string in the searchable part.
Out reference(const std::vector<std::string> & stops, bool think, const std::string & s) {
    size_t from = 0;
    if (think) {
        const size_t k = s.find("</think>");
        if (k == std::string::npos) return {s, false};
        from = k + 8;
    }
    size_t best = std::string::npos;
    for (const std::string & st : stops) {
        const size_t p = s.find(st, from);
        if (p != std::string::npos && p < best) best = p;
    }
    if (best == std::string::npos) return {s, false};
    return {s.substr(0, best), true};
}

int fails = 0, checks = 0;

void check(const char * name, bool ok, const std::string & detail = "") {
    ++checks;
    if (!ok) { ++fails; std::printf("FAIL %s %s\n", name, detail.c_str()); }
}

void sample(const std::vector<std::string> & stops, bool think, const std::string & s) {
    const Out ref = reference(stops, think, s);
    const Out one = run(stops, think, {s});
    check("one piece == reference", one.text == ref.text && one.hit == ref.hit, "[" + s + "] got [" + one.text + "]");
    for (size_t i = 0; i <= s.size(); ++i) {
        const Out two = run(stops, think, {s.substr(0, i), s.substr(i)});
        if (two.text != ref.text || two.hit != ref.hit) { check("2-split", false, "[" + s + "] at " + std::to_string(i)); return; }
        for (size_t j = i; j <= s.size(); ++j) {
            const Out three = run(stops, think, {s.substr(0, i), s.substr(i, j - i), s.substr(j)});
            if (three.text != ref.text || three.hit != ref.hit) { check("3-split", false, "[" + s + "] at " + std::to_string(i) + "," + std::to_string(j)); return; }
        }
    }
    std::vector<std::string> bytes;
    for (char c : s) bytes.emplace_back(1, c);
    const Out b = run(stops, think, bytes);
    check("byte-by-byte", b.text == ref.text && b.hit == ref.hit, "[" + s + "]");
    check("split-invariant", true);
}

}  // namespace

int main() {
    const std::vector<std::string> st1 = {"\nUser:", "###"};
    sample(st1, false, "Hello there.\nUser: next question");
    sample(st1, false, "no stop string here at all");
    sample(st1, false, "partial #a## then ## and finally ### end");
    sample(st1, false, "ends with a partial stop \nUse");
    sample(st1, false, "###at the very start");
    sample({"END"}, false, "日本語のテキスト END 以降");
    sample({"END", "EN"}, false, "abc EN then END");            // the earliest match wins, even if a longer stop starts there
    sample({"aab"}, false, "aaab");                               // overlapping prefix
    sample({"\n\n"}, true, "<think>\nreasoning\n\nmore\n</think>\n\nanswer line\n\nignored");   // stop inside the reasoning is not applied
    sample({"STOP"}, true, "<think>STOP is fine here</think>answer STOP rest");
    sample({"STOP"}, true, "<think>never closed STOP");
    sample({"x"}, true, "<think>a</thi");                          // cut tag at the end
    std::printf("stop_filter_test: %d checks, %d failed -> %s\n", checks, fails, fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
