"use client";

import { useState } from "react";
import { Copy, Check } from "lucide-react";

export default function CodeBlock({ code, lang = "" }) {
  const [copied, setCopied] = useState(false);
  return (
    <div className="relative">
      {lang && (
        <span className="absolute left-4 top-3 font-mono text-[10px] uppercase tracking-wider text-slate-500">
          {lang}
        </span>
      )}
      <button
        onClick={async () => {
          await navigator.clipboard.writeText(code);
          setCopied(true);
          setTimeout(() => setCopied(false), 1500);
        }}
        className="absolute right-3 top-3 grid h-8 w-8 place-items-center rounded-lg border border-white/10 bg-white/5 text-slate-300 transition hover:text-white"
        aria-label="Copy code"
      >
        {copied ? (
          <Check className="h-4 w-4 text-brand-emerald" />
        ) : (
          <Copy className="h-4 w-4" />
        )}
      </button>
      <pre className="overflow-x-auto rounded-xl border border-white/10 bg-ink-900/70 p-4 pt-9 font-mono text-xs leading-relaxed text-slate-200">
        <code>{code}</code>
      </pre>
    </div>
  );
}
