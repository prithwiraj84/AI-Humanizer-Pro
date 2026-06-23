import Link from "next/link";
import { Sparkles } from "lucide-react";

export default function AuthShell({ title, subtitle, children, footer }) {
  return (
    <main className="relative flex min-h-screen min-h-[100svh] items-center justify-center px-5 py-12">
      <div className="absolute inset-0 -z-10 bg-[radial-gradient(circle_at_50%_-10%,rgba(99,102,241,0.2),transparent_55%)]" />
      <div className="absolute inset-0 -z-10 bg-[radial-gradient(circle_at_80%_100%,rgba(34,211,238,0.12),transparent_45%)]" />

      <div className="w-full max-w-md">
        <Link
          href="/"
          className="mb-8 flex items-center justify-center gap-2 font-display text-xl font-bold"
        >
          <span className="grid h-10 w-10 place-items-center rounded-xl bg-brand-gradient shadow-glow">
            <Sparkles className="h-5 w-5 text-white" />
          </span>
          AI Humanizer<span className="gradient-text"> Pro</span>
        </Link>

        <div className="glass-strong rounded-2xl p-8 shadow-2xl">
          <h1 className="font-display text-2xl font-bold">{title}</h1>
          {subtitle && <p className="mt-1.5 text-sm text-slate-400">{subtitle}</p>}
          <div className="mt-6">{children}</div>
        </div>

        {footer && (
          <div className="mt-6 text-center text-sm text-slate-400">{footer}</div>
        )}
      </div>
    </main>
  );
}
