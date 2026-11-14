import pathlib
import tempfile
import unittest

import kiro_gh_agent as a

CFG = a.load_config()
ME = {"id": 68177813, "login": "mcpe500", "type": "User"}
IBRAHIM = {"id": 73787745, "login": "ibrahimmuh26", "type": "User"}
NICO = {"id": 113233053, "login": "nicostap", "type": "User"}
STRANGER = {"id": 999, "login": "stranger", "type": "User"}


def ev(user=ME, body="/kiro halo", kind="comment", created="2026-11-13T10:00:00Z", updated=None):
    return {"id": "c1", "kind": kind, "number": 1, "user": user, "body": body,
            "created_at": created, "updated_at": updated or created, "url": "u"}


class Allowlist(unittest.TestCase):
    def test_three_allowed(self):
        for u in (ME, IBRAHIM, NICO):
            self.assertEqual(a.classify(ev(user=u), CFG)[0], "run", u["login"])

    def test_stranger_ignored(self):
        self.assertEqual(a.classify(ev(user=STRANGER), CFG), ("skip", "pengirim tidak diizinkan"))

    def test_same_login_different_id_ignored(self):
        impostor = {"id": 12345, "login": "mcpe500", "type": "User"}
        self.assertEqual(a.classify(ev(user=impostor), CFG)[0], "skip")

    def test_bot_with_allowed_id_ignored(self):
        self.assertEqual(a.classify(ev(user={**ME, "type": "Bot"}), CFG)[0], "skip")

    def test_missing_user_ignored(self):
        self.assertEqual(a.classify(ev(user=None), CFG)[0], "skip")
        self.assertEqual(a.classify(ev(user={}), CFG)[0], "skip")

    def test_id_compared_as_string_not_substring(self):
        self.assertFalse(a.is_allowed({"id": 6817781, "type": "User"}, CFG))


class Trigger(unittest.TestCase):
    def test_requires_prefix(self):
        self.assertEqual(a.classify(ev(body="halo /kiro tes"), CFG)[1], "tanpa pemicu")
        self.assertEqual(a.classify(ev(body="> /kiro tes"), CFG)[1], "tanpa pemicu")

    def test_requires_word_boundary(self):
        self.assertEqual(a.classify(ev(body="/kirofoo"), CFG)[1], "tanpa pemicu")

    def test_instruction_extracted(self):
        self.assertEqual(a.classify(ev(body="/kiro  buat file\nbaris dua"), CFG),
                         ("run", "buat file\nbaris dua"))

    def test_empty_instruction_skipped(self):
        self.assertEqual(a.classify(ev(body="/kiro   "), CFG)[1], "instruksi kosong")

    def test_leading_whitespace_and_bom(self):
        self.assertEqual(a.parse_trigger("\ufeff \n/kiro x", "/kiro"), "x")

    def test_instruction_clipped(self):
        long = "/kiro " + "x" * 20000
        self.assertEqual(len(a.classify(ev(body=long), CFG)[1]), CFG["max_instruction_chars"])


class Loops(unittest.TestCase):
    def test_agent_comment_ignored_even_if_allowed_author(self):
        body = f"{a.MARKER}\n/kiro ulangi"
        self.assertEqual(a.classify(ev(body=body), CFG)[1], "komentar agen sendiri")

    def test_edited_comment_ignored(self):
        e = ev(updated="2026-11-13T10:05:00Z")
        self.assertEqual(a.classify(e, CFG)[1], "komentar sudah diedit")

    def test_issue_body_edit_rule_not_applied(self):
        e = ev(kind="issue", updated="2026-11-13T10:05:00Z")
        self.assertEqual(a.classify(e, CFG)[0], "run")


class Limits(unittest.TestCase):
    def test_rate_limit_12(self):
        now = 10_000.0
        self.assertTrue(a.rate_ok([now - 10] * 11, now, 12))
        self.assertFalse(a.rate_ok([now - 10] * 12, now, 12))

    def test_old_runs_dont_count(self):
        now = 10_000.0
        self.assertTrue(a.rate_ok([now - 3601] * 50, now, 12))

    def test_limit_value_in_config(self):
        self.assertEqual(CFG["max_runs_per_hour"], 12)


class Guards(unittest.TestCase):
    def test_protected_paths(self):
        paths = ["spec/a.md", "tools/agent/kiro_gh_agent.py", ".github/workflows/x.yml", ".kiro/agents/x.json", ".git/config"]
        self.assertEqual(a.violates_protected(paths, CFG["protected_paths"]), paths[1:])

    def test_similar_prefix_not_protected(self):
        self.assertEqual(a.violates_protected(["tools/agentx/a.py", "spec/.github/x"], CFG["protected_paths"]), [])

    def test_mentions_neutralized(self):
        out = a.neutralize_mentions("halo @octocat dan @a-b, email x@y.com")
        self.assertNotIn("@octocat", out)
        self.assertIn("@\u200boctocat", out)

    def test_safe_ref(self):
        for bad in ("-rf", "a b", "../x", "a;b", ""):
            self.assertFalse(a.SAFE_REF.match(bad) and ".." not in bad, bad)
        self.assertTrue(a.SAFE_REF.match("kiro/issue-1-c123"))


class Normalize(unittest.TestCase):
    def test_comment_number_from_issue_url(self):
        c = {"id": 5, "issue_url": "https://api.github.com/repos/o/r/issues/42", "user": {"id": 1, "login": "x", "type": "User"},
             "body": None, "created_at": "t", "updated_at": "t", "html_url": "h"}
        n = a.norm_comment(c)
        self.assertEqual((n["number"], n["body"], n["id"]), (42, "", "c5"))

    def test_deleted_user_does_not_crash(self):
        c = {"id": 5, "issue_url": "https://x/issues/1", "user": None, "body": "/kiro x",
             "created_at": "t", "updated_at": "t", "html_url": "h"}
        self.assertEqual(a.classify(a.norm_comment(c), CFG)[0], "skip")


def pr(number, ref, repo="mcpe500/legacy1070qi38"):
    return {"number": number, "head": {"ref": ref, "repo": {"full_name": repo}}}


class SinglePr(unittest.TestCase):
    R = "mcpe500/legacy1070qi38"

    def test_follow_up_extends_existing_pr(self):
        prs = [pr(18, "kiro/issue-1-c6875260454")]
        self.assertEqual(a.pick_reusable_pr(prs, 1, self.R), {"number": 18, "ref": "kiro/issue-1-c6875260454"})

    def test_no_pr_returns_none(self):
        self.assertIsNone(a.pick_reusable_pr([], 1, self.R))

    def test_issue_1_does_not_take_issue_12(self):
        self.assertIsNone(a.pick_reusable_pr([pr(30, "kiro/issue-12-c1"), pr(31, "kiro/issue-100")], 1, self.R))

    def test_exact_branch_name_matches(self):
        self.assertEqual(a.pick_reusable_pr([pr(5, "kiro/issue-7")], 7, self.R)["number"], 5)

    def test_fork_pr_never_reused(self):
        self.assertIsNone(a.pick_reusable_pr([pr(9, "kiro/issue-1-x", repo="evil/fork")], 1, self.R))

    def test_deleted_fork_repo_never_reused(self):
        self.assertIsNone(a.pick_reusable_pr([{"number": 9, "head": {"ref": "kiro/issue-1-x", "repo": None}}], 1, self.R))

    def test_non_kiro_branch_ignored(self):
        self.assertIsNone(a.pick_reusable_pr([pr(9, "feature/issue-1")], 1, self.R))

    def test_newest_wins(self):
        prs = [pr(18, "kiro/issue-1-a"), pr(25, "kiro/issue-1-b")]
        self.assertEqual(a.pick_reusable_pr(prs, 1, self.R)["number"], 25)


class Visuals(unittest.TestCase):
    def test_no_svg_no_section(self):
        self.assertEqual(a.visual_section("o/r", "abc123", []), "")

    def test_permalink_uses_sha_and_raw(self):
        out = a.visual_section("o/r", "abc123", ["spec/specs/assets/a b.svg"])
        self.assertIn("/blob/abc123/spec/specs/assets/a%20b.svg?raw=true", out)
        self.assertIn("![a b]", out)
        self.assertTrue(out.startswith("## Gambaran visual"))

    def test_each_svg_embedded(self):
        out = a.visual_section("o/r", "s", ["a.svg", "b.svg"])
        self.assertEqual(out.count("?raw=true"), 2)


class SvgChecks(unittest.TestCase):
    def write(self, text, name="x.svg"):
        d = tempfile.mkdtemp()
        p = pathlib.Path(d) / name
        p.write_text(text, encoding="utf-8")
        return p

    GOOD = ('<svg xmlns="http://www.w3.org/2000/svg" width="400" height="100" viewBox="0 0 400 100">'
            '<rect x="10" y="10" width="380" height="80"/><text x="20" y="40" font-size="14">halo</text></svg>')

    def test_good_svg(self):
        self.assertIsNone(a.svg_error(self.write(self.GOOD)))

    def test_broken_xml(self):
        self.assertIn("XML", a.svg_error(self.write("<svg><text></svg>")))

    def test_script_rejected(self):
        bad = self.GOOD.replace("</svg>", "<script>alert(1)</script></svg>")
        self.assertIn("script", a.svg_error(self.write(bad)))

    def test_event_handler_rejected(self):
        bad = self.GOOD.replace("<rect ", '<rect onload="x()" ')
        self.assertIsNotNone(a.svg_error(self.write(bad)))

    def test_entity_rejected(self):
        bad = '<!DOCTYPE svg [<!ENTITY a "b">]>' + self.GOOD
        self.assertIsNotNone(a.svg_error(self.write(bad)))

    def test_not_svg_root(self):
        self.assertIn("bukan <svg>", a.svg_error(self.write("<html/>")))

    def test_missing_file(self):
        self.assertIsNotNone(a.svg_error("/nonexistent/x.svg"))

    def test_too_big(self):
        self.assertIn("500 KB", a.svg_error(self.write(self.GOOD + " " * 600_000)))

    def test_layout_ok(self):
        self.assertEqual(a.svg_layout_warnings(self.write(self.GOOD)), [])

    def test_layout_text_leaves_box(self):
        bad = self.GOOD.replace(">halo<", ">" + "x" * 80 + "<")
        self.assertTrue(any("kotak" in w or "kanvas" in w for w in a.svg_layout_warnings(self.write(bad))))

    def test_layout_tiny_font(self):
        bad = self.GOOD.replace('font-size="14"', 'font-size="8"')
        self.assertTrue(any("huruf kecil" in w for w in a.svg_layout_warnings(self.write(bad))))

    def test_layout_text_outside_canvas(self):
        bad = self.GOOD.replace('x="20" y="40"', 'x="390" y="40"').replace(">halo<", ">panjang sekali<")
        self.assertTrue(any("kanvas" in w for w in a.svg_layout_warnings(self.write(bad))))


class Intent(unittest.TestCase):
    def test_kind(self):
        self.assertEqual(a.work_kind(True, 18), "pr")
        self.assertEqual(a.work_kind(False, 18), "issue_with_pr")
        self.assertEqual(a.work_kind(False, None), "issue_new")

    def test_comment_in_pr_never_says_new_pr(self):
        msg = a.start_message("mcpe500", "pr", "tanpa shell", 18)
        self.assertNotIn("membuat PR baru", msg)
        self.assertIn("PR ini (#18)", msg)
        self.assertIn("branch PR ini", msg)

    def test_issue_new_says_analyze_first_and_no_pr_if_unneeded(self):
        msg = a.start_message("mcpe500", "issue_new", "tanpa shell")
        self.assertIn("analisis dulu", msg)
        self.assertIn("tanpa membuat PR", msg)
        self.assertNotIn("membuat PR baru bila ada perubahan", msg)

    def test_issue_with_pr_points_to_existing_pr(self):
        msg = a.start_message("mcpe500", "issue_with_pr", "tanpa shell", 21)
        self.assertIn("PR #21", msg)
        self.assertIn("bukan PR baru", msg)

    def test_no_message_promises_a_pr_unconditionally(self):
        for kind, n in (("pr", 5), ("issue_with_pr", 5), ("issue_new", None)):
            self.assertNotIn("bila ada perubahan)", a.start_message("x", kind, "m", n))

    def test_login_in_message(self):
        self.assertIn("@ibrahimmuh26", a.start_message("ibrahimmuh26", "issue_new", "m"))

    def test_decision_conflict_blocks_pr(self):
        s = "Keputusan: tidak perlu PR. Hanya menjawab.\nTapi saya mengubah berkas."
        self.assertTrue(a.decision_conflict("issue_new", s, ["spec/a.md"]))

    def test_decision_ok_when_no_changes(self):
        self.assertFalse(a.decision_conflict("issue_new", "Keputusan: tidak perlu PR.", []))

    def test_need_pr_decision_is_not_a_conflict(self):
        self.assertFalse(a.decision_conflict("issue_new", "Keputusan: perlu PR. Menulis berkas.", ["a.md"]))

    def test_decision_phrase_variants(self):
        for s in ("KEPUTUSAN:  TIDAK   PERLU PR", "keputusan: tidak perlu pr"):
            self.assertTrue(a.decision_conflict("issue_new", s, ["a"]))

    def test_conflict_only_applies_to_fresh_issue(self):
        s = "Keputusan: tidak perlu PR"
        self.assertFalse(a.decision_conflict("pr", s, ["a"]))
        self.assertFalse(a.decision_conflict("issue_with_pr", s, ["a"]))

    def test_none_summary_safe(self):
        self.assertFalse(a.decision_conflict("issue_new", None, ["a"]))

    def test_prompt_rules_per_kind(self):
        self.assertIn("analisis dulu", a.scope_rules("issue_new"))
        self.assertIn("JANGAN mengubah file", a.scope_rules("issue_new"))
        self.assertIn("Keputusan: perlu PR", a.scope_rules("issue_new"))
        self.assertIn("PR #18", a.scope_rules("pr", 18))
        self.assertIn("Jangan menyarankan PR baru", a.scope_rules("pr", 18))
        self.assertIn("PR #21", a.scope_rules("issue_with_pr", 21))
        self.assertNotIn("Keputusan:", a.scope_rules("pr", 18))

    def test_build_prompt_contains_scope_for_each_kind(self):
        info = {"title": "t", "body": "b", "author": {"id": 68177813, "login": "mcpe500", "type": "User"}}
        e = ev()
        for kind, n, needle in (("pr", 18, "di dalam PR #18"), ("issue_new", None, "belum punya PR")):
            p = a.build_prompt(CFG, e, "tanya saja", info, "x", False, kind, n)
            self.assertIn(needle, p)
            self.assertIn("=== INSTRUKSI (dari @mcpe500) ===", p)


class SequentialThinking(unittest.TestCase):
    def cfg(self, node=True, entry=True):
        d = tempfile.mkdtemp()
        n, e = pathlib.Path(d) / "node", pathlib.Path(d) / "index.js"
        if node:
            n.write_text("x")
        if entry:
            e.write_text("x")
        return {**CFG, "sequential_thinking": {"server_name": "sequential-thinking", "node": str(n), "entry": str(e)}}

    def test_ready_only_when_both_files_exist(self):
        self.assertTrue(a.sequential_ready(self.cfg()))
        self.assertFalse(a.sequential_ready(self.cfg(node=False)))
        self.assertFalse(a.sequential_ready(self.cfg(entry=False)))

    def test_not_configured_means_not_ready(self):
        c = {k: v for k, v in CFG.items() if k != "sequential_thinking"}
        self.assertFalse(a.sequential_ready(c))
        self.assertIsNone(a.sequential_paths(c))

    def test_home_is_expanded(self):
        sp = a.sequential_paths(CFG)
        self.assertFalse(sp["node"].startswith("~"))
        self.assertFalse(sp["entry"].startswith("~"))

    def test_profile_registers_mcp_when_ready(self):
        for shell in (False, True):
            p = a.profile(self.cfg(), shell)
            self.assertIn("sequential-thinking", p["mcpServers"])
            self.assertIn("@sequential-thinking", p["tools"])
            self.assertIn({"capability": "mcp", "match": ["sequential-thinking/*"], "effect": "allow"},
                          p["permissions"]["rules"])
            self.assertEqual(p["mcpServers"]["sequential-thinking"]["env"], {"DISABLE_THOUGHT_LOGGING": "true"})

    def test_profile_without_mcp_when_not_ready(self):
        p = a.profile(self.cfg(entry=False), False)
        self.assertNotIn("mcpServers", p)
        self.assertNotIn("@sequential-thinking", p["tools"])
        self.assertFalse([r for r in p["permissions"]["rules"] if r["capability"] == "mcp"])

    def test_mcp_permission_is_scoped_to_that_server_only(self):
        rules = [r for r in a.profile(self.cfg(), False)["permissions"]["rules"] if r["capability"] == "mcp"]
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0]["match"], ["sequential-thinking/*"])

    def test_shell_still_denied_in_noshell_profile(self):
        rules = a.profile(self.cfg(), False)["permissions"]["rules"]
        self.assertIn({"capability": "shell", "effect": "deny"}, rules)

    def test_thinking_rules_ready(self):
        t = a.thinking_rules(True, "mcpe500")
        self.assertIn("sequentialthinking", t)
        self.assertIn("@mcpe500", t)
        self.assertIn("Pemahaman", t)
        self.assertIn("perlu mengubah", t)

    def test_thinking_rules_not_ready_says_so(self):
        t = a.thinking_rules(False, "mcpe500")
        self.assertIn("TIDAK tersedia", t)
        self.assertNotIn("Pemahaman", t)

    def test_prompt_includes_thinking_rule(self):
        info = {"title": "t", "body": "b", "author": {"id": 68177813, "login": "mcpe500", "type": "User"}}
        ready = a.build_prompt(self.cfg(), ev(), "x", info, "d", False)
        notready = a.build_prompt(self.cfg(entry=False), ev(), "x", info, "d", False)
        self.assertIn("Pemahaman", ready)
        self.assertIn("TIDAK tersedia", notready)

    def test_prompt_keeps_injection_boundary(self):
        info = {"title": "t", "body": "b", "author": STRANGER}
        p = a.build_prompt(self.cfg(), ev(), "x", info, "d", False)
        self.assertIn("sengaja tidak diberikan", p)
        self.assertIn("=== INSTRUKSI (dari @mcpe500) ===", p)

    def test_install_script_pins_versions(self):
        src = (pathlib.Path(a.HERE) / "install-sequential-thinking.sh").read_text()
        self.assertIn("--ignore-scripts", src)
        self.assertIn("sha256sum -c", src)
        self.assertIn('NODE_VERSION="${NODE_VERSION:-v24', src)
        self.assertRegex(src, r'MCP_VERSION="\$\{MCP_VERSION:-\d+\.\d+\.\d+\}"')
        self.assertNotIn("sudo ", src.replace("tanpa sudo", "").replace("Tanpa sudo", ""))


if __name__ == "__main__":
    unittest.main()
