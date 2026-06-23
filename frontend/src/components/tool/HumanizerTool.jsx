"use client";

import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import {
  Wand2,
  Loader2,
  Copy,
  Check,
  Eraser,
  Zap,
  GitCompareArrows,
} from "lucide-react";
import Navbar from "@/components/layout/Navbar";
import { api } from "@/lib/api";

const TONES = [
  { value: "standard", label: "Standard" },
  { value: "professional", label: "Professional / Formal" },
  { value: "casual", label: "Casual / Conversational" },
  { value: "academic", label: "Academic / Complex" },
];

// Render the backend diff HTML safely: parse it and keep ONLY the known
// span.added / span.unchanged structure with DOM-escaped text. Any injected
// markup degrades to plain text — no XSS, and literal <>/code render correctly.
function renderSafeDiff(html) {
  if (typeof window === "undefined" || !html) return null;
  const doc = new DOMParser().parseFromString(html, "text/html");
  const nodes = [];
  doc.body.childNodes.forEach((node, i) => {
    if (node.nodeType === Node.TEXT_NODE) {
      nodes.push(node.textContent);
    } else if (node.nodeType === Node.ELEMENT_NODE && node.tagName === "SPAN") {
      const cls = node.classList.contains("added")
        ? "added"
        : node.classList.contains("unchanged")
        ? "unchanged"
        : undefined;
      nodes.push(
        <span key={i} className={cls}>
          {node.textContent}
        </span>
      );
    } else {
      nodes.push(node.textContent);
    }
  });
  return nodes;
}

function Metric({ label, value, tone = "indigo" }) {
  const tones = {
    indigo: "text-indigo-300",
    emerald: "text-emerald-300",
    cyan: "text-cyan-300",
    pink: "text-pink-300",
  };
  return (
    <div className="glass rounded-xl p-4 text-center">
      <div className={`font-display text-2xl font-bold ${tones[tone]}`}>
        {value}
      </div>
      <div className="mt-1 text-xs text-slate-400">{label}</div>
    </div>
  );
}

export default function HumanizerTool() {
  const router = useRouter();
  const [text, setText] = useState("");
  const [tone, setTone] = useState("standard");
  const [deepMode, setDeepMode] = useState(false);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");
  const [result, setResult] = useState(null);
  const [copied, setCopied] = useState(false);
  const [showDiff, setShowDiff] = useState(false);

  const charCount = text.length;
  const canSubmit = useMemo(() => text.trim().length > 0 && !loading, [text, loading]);

  async function humanize() {
    setErr("");
    setLoading(true);
    setResult(null);
    try {
      const data = await api.humanize({ text, tone, deep_mode: deepMode });
      setResult(data);
    } catch (e) {
      if (e.status === 401) {
        router.push("/login");
        return;
      }
      setErr(e.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  async function copyOut() {
    if (!result?.humanized) return;
    await navigator.clipboard.writeText(result.humanized);
    setCopied(true);
    setTimeout(() => setCopied(false), 1600);
  }

  return (
    <>
      <Navbar />
      <main className="container-x px-5 pb-24 pt-28 sm:px-8">
        <div className="mb-8">
          <h1 className="font-display text-3xl font-bold sm:text-4xl">
            AI <span className="gradient-text">Humanizer</span>
          </h1>
          <p className="mt-2 text-slate-400">
            Paste your AI-generated text, pick a tone, and humanize it instantly.
          </p>
        </div>

        <div className="grid gap-6 lg:grid-cols-2">
          {/* Input */}
          <section className="glass-strong rounded-2xl p-6">
            <div className="flex flex-wrap items-center gap-3">
              <select
                value={tone}
                onChange={(e) => setTone(e.target.value)}
                aria-label="Tone"
                className="field max-w-[16rem]"
              >
                {TONES.map((t) => (
                  <option key={t.value} value={t.value} className="bg-ink-800">
                    {t.label}
                  </option>
                ))}
              </select>

              <button
                type="button"
                onClick={() => setDeepMode((v) => !v)}
                aria-pressed={deepMode}
                className={`btn gap-2 border ${
                  deepMode
                    ? "border-brand-violet/50 bg-brand-violet/15 text-violet-200"
                    : "border-white/10 bg-white/5 text-slate-300"
                }`}
              >
                <Zap className="h-4 w-4" />
                Deep mode {deepMode ? "ON" : "OFF"}
              </button>
            </div>

            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Paste your AI-generated text here…"
              aria-label="Text to humanize"
              className="field mt-4 h-72 resize-none leading-relaxed"
            />

            <div className="mt-3 flex items-center justify-between">
              <span className="text-xs text-slate-400">{charCount} characters</span>
              <div className="flex gap-2">
                <button
                  onClick={() => {
                    setText("");
                    setResult(null);
                    setErr("");
                  }}
                  className="btn-ghost"
                >
                  <Eraser className="h-4 w-4" />
                  Clear
                </button>
                <button onClick={humanize} disabled={!canSubmit} className="btn-primary">
                  {loading ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Wand2 className="h-4 w-4" />
                  )}
                  Humanize
                </button>
              </div>
            </div>

            {err && (
              <div className="mt-4 rounded-lg border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-300">
                {err}
              </div>
            )}
          </section>

          {/* Output */}
          <section className="glass-strong rounded-2xl p-6">
            {!result ? (
              <div className="flex h-full min-h-[20rem] flex-col items-center justify-center text-center text-slate-400">
                <Wand2 className="h-10 w-10 text-slate-600" />
                <p className="mt-3 text-sm">
                  Your humanized text and analysis will appear here.
                </p>
              </div>
            ) : (
              <div className="space-y-5">
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                  <Metric label="AI probability" value={result.ai_probability} tone="emerald" />
                  <Metric label="Confidence" value={result.confidence} tone="cyan" />
                  <Metric label="Structural change" value={result.structural_changes} tone="indigo" />
                  <Metric label="Words changed" value={result.changed_words} tone="pink" />
                </div>

                <div className="flex items-center justify-between">
                  <span className="chip">
                    <Check className="h-3.5 w-3.5 text-brand-emerald" />
                    {result.classification}
                  </span>
                  <div className="flex gap-2">
                    <button
                      onClick={() => setShowDiff((v) => !v)}
                      className="btn-ghost"
                    >
                      <GitCompareArrows className="h-4 w-4" />
                      {showDiff ? "Hide diff" : "Show diff"}
                    </button>
                    <button onClick={copyOut} className="btn-ghost">
                      {copied ? (
                        <Check className="h-4 w-4 text-brand-emerald" />
                      ) : (
                        <Copy className="h-4 w-4" />
                      )}
                      {copied ? "Copied" : "Copy"}
                    </button>
                  </div>
                </div>

                {showDiff ? (
                  <div className="diff-output max-h-72 overflow-auto rounded-xl border border-white/10 bg-ink-900/60 p-4 text-sm leading-relaxed">
                    {renderSafeDiff(result.diff_html)}
                  </div>
                ) : (
                  <div className="max-h-72 overflow-auto whitespace-pre-wrap rounded-xl border border-white/10 bg-ink-900/60 p-4 text-sm leading-relaxed text-slate-100">
                    {result.humanized}
                  </div>
                )}

                {result.longest_unchanged && (
                  <p className="text-xs text-slate-400">
                    Longest unchanged phrase:{" "}
                    <span className="text-slate-400">{result.longest_unchanged}</span>
                  </p>
                )}
              </div>
            )}
          </section>
        </div>
      </main>
    </>
  );
}
